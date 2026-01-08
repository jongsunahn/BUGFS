#!/usr/bin/env python3
"""
MCP server that crawls GitHub issues, embeds the reports, and recommends bug
locations by reusing the crawl/embedding utilities plus the RQ1 workflow.

Uses sqlite-vec for efficient vector similarity search with minimal memory footprint.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import struct
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import sqlite_vec
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from crawl_issues_With_diffs import crawl_issues_with_diffs

try:
    from fastmcp import FastMCP, Context
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "The modelcontextprotocol package is required. "
        "Install it with `pip install modelcontextprotocol`."
    ) from exc


load_dotenv()

# Configure logging
LOG_DIR = Path(os.getenv("BUGFS_LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "bugfs_mcp.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("bugfs_mcp")


DEFAULT_MODEL = os.getenv("BUGFS_EMBED_MODEL", "text-embedding-3-small")
CHUNK_CHAR_LIMIT = int(os.getenv("BUGFS_CHUNK_CHAR_LIMIT", "3000"))
DATA_DIR = Path(os.getenv("BUGFS_DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Embedding dimension for text-embedding-3-small
EMBEDDING_DIM = 1536

# Regex to extract hunk header: line info and context
HUNK_HEADER_REGEX = re.compile(
    r"@@ -\d+,\d+ \+(?P<start>\d+),(?P<length>\d+) @@(?:\s+(?P<context>.+))?$",
    flags=re.MULTILINE,
)

# Patterns to extract method/function name from the hunk context
# Python: def function_name
PYTHON_FUNC_PATTERN = re.compile(r"def\s+(?P<name>\w+)")

# Java/Groovy/Kotlin method: modifiers + return_type + method_name(
# e.g., "public void methodName(", "private CloseableHttpClient create("
JAVA_METHOD_PATTERN = re.compile(
    r"(?:public|private|protected|static|final|abstract|synchronized|void|\w+)\s+(?P<name>\w+)\s*\("
)

# Class/interface/enum definition
CLASS_PATTERN = re.compile(r"(?:class|interface|enum)\s+(?P<name>\w+)")


def slugify_repo(owner: str, repo: str) -> str:
    return f"{owner}_{repo}".replace("/", "_")


def ensure_text(value: Optional[str]) -> str:
    return (value or "").strip()


def build_issue_text(issue: Dict[str, Any]) -> str:
    title = ensure_text(issue.get("title"))
    body = ensure_text(issue.get("body"))
    text = f"{title}\n\n{body}".strip()
    return text


def split_into_chunks(text: str, limit: int = CHUNK_CHAR_LIMIT) -> List[str]:
    if not text:
        return []
    return [text[i : i + limit] for i in range(0, len(text), limit)]


def sanitize_issue_body(issue: Dict[str, Any], min_word_count: int = 1) -> Optional[str]:
    text = build_issue_text(issue)
    text = re.sub(r"\s+", " ", text)
    if len(text.split()) < min_word_count:
        return None
    return text


def normalize_filename(path_str: str) -> str:
    path_str = path_str.replace("\\", "/")
    parts = [p for p in path_str.split("/") if p]
    if not parts:
        return path_str
    return parts[-1]


@dataclass(frozen=True)
class IssueLocation:
    start: int
    length: int
    function: str
    file: str

    @property
    def label(self) -> str:
        return f"{self.file}:{self.function}"


def extract_method_from_context(context: str) -> Optional[str]:
    """Extract method/function name from git diff hunk context."""
    if not context:
        return None

    # Try Python function pattern
    match = PYTHON_FUNC_PATTERN.search(context)
    if match:
        return match.group("name")

    # Try Java/Groovy/Kotlin method pattern
    match = JAVA_METHOD_PATTERN.search(context)
    if match:
        name = match.group("name")
        # Filter out common false positives (keywords that look like method names)
        if name not in ("if", "for", "while", "switch", "catch", "synchronized", "return", "new", "throw"):
            return name

    # Try class/interface/enum pattern
    match = CLASS_PATTERN.search(context)
    if match:
        return match.group("name")

    return None


def extract_locations(issue: Dict[str, Any]) -> List[IssueLocation]:
    """Extract file:function locations from git diff patches.

    Parses the hunk header context (text after @@ ... @@) to find the enclosing
    function/method name. Supports Python, Java, Groovy, Kotlin, and class definitions.
    """
    locations: List[IssueLocation] = []
    seen: set[tuple[str, str]] = set()  # Avoid duplicates

    for commit in issue.get("commits", []):
        filename = normalize_filename(commit.get("filename", "unknown"))
        patch = commit.get("patch")
        if not patch:
            continue

        # Parse each hunk header
        for match in HUNK_HEADER_REGEX.finditer(patch):
            start = int(match.group("start"))
            length = int(match.group("length"))
            context = match.group("context") or ""

            # Try to extract method/function name from context
            method_name = extract_method_from_context(context)

            if method_name:
                key = (filename, method_name)
                if key not in seen:
                    seen.add(key)
                    locations.append(
                        IssueLocation(start=start, length=length, function=method_name, file=filename)
                    )

    return locations


class Embedder:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set. Please configure your OpenAI API key.")
            self._client = OpenAI(api_key=api_key)
        return self._client

    def embed_text(self, text: str) -> np.ndarray:
        chunks = split_into_chunks(text)
        if not chunks:
            raise ValueError("Cannot embed empty text.")
        last_error: Optional[Exception] = None
        for attempt in range(5):
            try:
                response = self.client.embeddings.create(model=self.model, input=chunks)
                vectors = [np.array(item.embedding, dtype=np.float32) for item in response.data]
                avg_vector = np.mean(vectors, axis=0)
                return avg_vector.astype(np.float32)
            except Exception as exc:
                time.sleep(1 + attempt)
                last_error = exc
        raise RuntimeError(f"Failed to embed text after retries: {last_error}")


def serialize_f32(vector: np.ndarray) -> bytes:
    """Serialize a numpy float32 vector to bytes for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize vector for cosine similarity."""
    norm = np.linalg.norm(vector)
    if norm > 0:
        return vector / norm
    return vector


class SqliteVecCorpus:
    """Efficient vector search using sqlite-vec with minimal memory footprint."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def _init_schema(self):
        """Initialize database schema."""
        self.conn.executescript(f"""
            -- Raw issues table (stores full issue data as JSON)
            CREATE TABLE IF NOT EXISTS raw_issues (
                id INTEGER PRIMARY KEY,
                issue_id TEXT UNIQUE,
                data TEXT NOT NULL,
                created_at TEXT,
                updated_at TEXT
            );

            -- Indexed issues table (for vector search)
            CREATE TABLE IF NOT EXISTS issues (
                id INTEGER PRIMARY KEY,
                issue_id TEXT UNIQUE,
                content TEXT,
                locations TEXT
            );

            -- Vector index
            CREATE VIRTUAL TABLE IF NOT EXISTS issue_vectors USING vec0(
                id INTEGER PRIMARY KEY,
                embedding float[{EMBEDDING_DIM}]
            );

            CREATE INDEX IF NOT EXISTS idx_issue_id ON issues(issue_id);
            CREATE INDEX IF NOT EXISTS idx_raw_issue_id ON raw_issues(issue_id);
        """)
        self.conn.commit()

    def count(self) -> int:
        """Get number of indexed issues."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM issues")
        return cursor.fetchone()[0]

    def count_raw_issues(self) -> int:
        """Get number of raw issues stored."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM raw_issues")
        return cursor.fetchone()[0]

    def save_raw_issues(self, issues: Sequence[Dict[str, Any]]) -> int:
        """Save raw issues to database. Returns number of issues saved."""
        cursor = self.conn.cursor()
        saved = 0

        for issue in tqdm(issues, desc="Saving raw issues", unit="issue"):
            issue_id = str(issue.get("id", ""))
            if not issue_id:
                continue

            created_at = issue.get("created_at")
            updated_at = issue.get("updated_at")

            cursor.execute(
                """INSERT OR REPLACE INTO raw_issues (issue_id, data, created_at, updated_at)
                   VALUES (?, ?, ?, ?)""",
                (issue_id, json.dumps(issue, ensure_ascii=False), created_at, updated_at)
            )
            saved += 1

            if saved % 1000 == 0:
                self.conn.commit()

        self.conn.commit()
        return saved

    def load_raw_issues(self) -> List[Dict[str, Any]]:
        """Load all raw issues from database."""
        cursor = self.conn.execute("SELECT data FROM raw_issues")
        return [json.loads(row[0]) for row in cursor.fetchall()]

    def get_raw_issue(self, issue_id: str) -> Optional[Dict[str, Any]]:
        """Get a single raw issue by ID."""
        cursor = self.conn.execute(
            "SELECT data FROM raw_issues WHERE issue_id = ?", (issue_id,)
        )
        row = cursor.fetchone()
        return json.loads(row[0]) if row else None

    def get_raw_issue_ids(self) -> set:
        """Get set of all raw issue IDs."""
        cursor = self.conn.execute("SELECT issue_id FROM raw_issues")
        return {row[0] for row in cursor.fetchall()}

    def save_embedding(self, issue_id: str, embedding: List[float], model: str) -> None:
        """Save embedding for an issue."""
        cursor = self.conn.cursor()
        # Get existing data, update it, and save back
        cursor.execute("SELECT data FROM raw_issues WHERE issue_id = ?", (issue_id,))
        row = cursor.fetchone()
        if row:
            data = json.loads(row[0])
            data["embedding"] = embedding
            data["embedding_model"] = model
            cursor.execute(
                "UPDATE raw_issues SET data = ? WHERE issue_id = ?",
                (json.dumps(data, ensure_ascii=False), issue_id)
            )
            self.conn.commit()

    def get_embedded_issue_ids(self) -> set:
        """Get set of issue IDs that have embeddings."""
        cursor = self.conn.execute(
            "SELECT issue_id FROM raw_issues WHERE json_extract(data, '$.embedding') IS NOT NULL"
        )
        return {row[0] for row in cursor.fetchall()}

    def iter_raw_issues(self, batch_size: int = 100):
        """Iterate over raw issues in batches to reduce memory usage."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM raw_issues")
        total = cursor.fetchone()[0]

        for offset in range(0, total, batch_size):
            cursor = self.conn.execute(
                "SELECT issue_id, data FROM raw_issues LIMIT ? OFFSET ?",
                (batch_size, offset)
            )
            for row in cursor.fetchall():
                yield row[0], json.loads(row[1])

    def rebuild_vector_index(self, min_word_count: int = 1) -> int:
        """Rebuild vector index from raw_issues with embeddings."""
        # Clear existing index
        self.conn.execute("DELETE FROM issues")
        self.conn.execute("DELETE FROM issue_vectors")
        self.conn.commit()

        batch_issues = []
        batch_vectors = []
        indexed = 0

        for issue_id, issue in tqdm(
            self.iter_raw_issues(batch_size=500),
            desc="Building vector index",
            unit="issue",
            total=self.count_raw_issues()
        ):
            embedding = issue.get("embedding")
            if not embedding:
                continue

            # Handle double-encoded JSON strings
            if isinstance(embedding, str):
                try:
                    embedding = json.loads(embedding)
                except json.JSONDecodeError:
                    continue

            locations = extract_locations(issue)
            if not locations:
                continue
            text = sanitize_issue_body(issue, min_word_count=min_word_count)
            if not text:
                continue

            locations_as_dicts = [
                {"start": loc.start, "length": loc.length, "function": loc.function, "file": loc.file}
                for loc in locations
            ]

            vec = normalize_vector(np.asarray(embedding, dtype=np.float32))

            batch_issues.append((issue_id, text, json.dumps(locations_as_dicts)))
            batch_vectors.append(serialize_f32(vec))
            indexed += 1

            if len(batch_issues) >= 1000:
                self._batch_insert(batch_issues, batch_vectors)
                batch_issues = []
                batch_vectors = []

        if batch_issues:
            self._batch_insert(batch_issues, batch_vectors)

        return indexed

    @classmethod
    def build_from_issues(
        cls, db_path: Path, issues: Sequence[Dict[str, Any]], min_word_count: int = 1
    ) -> "SqliteVecCorpus":
        """Build sqlite-vec index from issues with embeddings."""
        # Remove existing db if exists
        if db_path.exists():
            db_path.unlink()

        corpus = cls(db_path)
        corpus._init_schema()

        batch_issues = []
        batch_vectors = []

        for issue in tqdm(issues, desc="Indexing issues", unit="issue"):
            embedding = issue.get("embedding")
            if not embedding:
                continue
            locations = extract_locations(issue)
            if not locations:
                continue
            text = sanitize_issue_body(issue, min_word_count=min_word_count)
            if not text:
                continue

            locations_as_dicts = [
                {"start": loc.start, "length": loc.length, "function": loc.function, "file": loc.file}
                for loc in locations
            ]

            # Normalize embedding for cosine similarity
            vec = normalize_vector(np.asarray(embedding, dtype=np.float32))

            batch_issues.append((
                str(issue["id"]),
                text,
                json.dumps(locations_as_dicts)
            ))
            batch_vectors.append(serialize_f32(vec))

            # Batch insert every 1000 records
            if len(batch_issues) >= 1000:
                corpus._batch_insert(batch_issues, batch_vectors)
                batch_issues = []
                batch_vectors = []

        # Insert remaining
        if batch_issues:
            corpus._batch_insert(batch_issues, batch_vectors)

        return corpus

    def _batch_insert(self, issues: List[Tuple], vectors: List[bytes]):
        """Batch insert issues and vectors."""
        cursor = self.conn.cursor()

        # Insert issues
        cursor.executemany(
            "INSERT OR REPLACE INTO issues (issue_id, content, locations) VALUES (?, ?, ?)",
            issues
        )

        # Get the rowids
        issue_ids = [i[0] for i in issues]
        placeholders = ",".join("?" * len(issue_ids))
        cursor.execute(
            f"SELECT id, issue_id FROM issues WHERE issue_id IN ({placeholders})",
            issue_ids
        )
        id_map = {row[1]: row[0] for row in cursor.fetchall()}

        # Insert vectors
        vector_data = [(id_map[issues[i][0]], vectors[i]) for i in range(len(issues))]
        cursor.executemany(
            "INSERT OR REPLACE INTO issue_vectors (id, embedding) VALUES (?, ?)",
            vector_data
        )

        self.conn.commit()

    def add_issue(self, issue_id: str, content: str, locations: List[Dict], embedding: np.ndarray):
        """Add a single issue to the index."""
        vec = normalize_vector(embedding.astype(np.float32))

        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO issues (issue_id, content, locations) VALUES (?, ?, ?)",
            (issue_id, content, json.dumps(locations))
        )
        rowid = cursor.lastrowid

        cursor.execute(
            "INSERT OR REPLACE INTO issue_vectors (id, embedding) VALUES (?, ?)",
            (rowid, serialize_f32(vec))
        )
        self.conn.commit()

    def recommend(
        self,
        query_vector: np.ndarray,
        neighbor_count: int = 10,
        max_candidates: int = 10,
    ) -> Dict[str, Any]:
        if self.count() == 0:
            raise RuntimeError("No issues with embeddings and locations are available.")

        # Normalize query vector for cosine similarity
        query = normalize_vector(query_vector.astype(np.float32))
        query_bytes = serialize_f32(query)

        # Search using sqlite-vec
        cursor = self.conn.execute("""
            SELECT
                i.issue_id,
                i.locations,
                v.distance
            FROM issue_vectors v
            JOIN issues i ON i.id = v.id
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
        """, (query_bytes, neighbor_count))

        neighbors: List[Dict[str, Any]] = []
        location_votes: Counter[str] = Counter()
        location_sources: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for row in cursor.fetchall():
            issue_id, locations_json, distance = row
            # sqlite-vec returns L2 distance, convert to similarity
            # For normalized vectors: cosine_similarity = 1 - (distance^2 / 2)
            similarity = float(1 - (distance ** 2) / 2)

            neighbors.append({"issue_id": issue_id, "similarity": similarity})

            locations = json.loads(locations_json)
            for loc in locations:
                label = f"{loc['file']}:{loc['function']}"
                location_votes[label] += 1
                location_sources[label].append({
                    "issue_id": issue_id,
                    "file": loc["file"],
                    "function": loc["function"],
                    "start": loc["start"],
                    "length": loc["length"],
                    "similarity": similarity,
                })

        recommendations: List[Dict[str, Any]] = []
        for label, votes in sorted(location_votes.items(), key=lambda item: (-item[1], item[0]))[:max_candidates]:
            sources = location_sources[label][:5]
            recommendations.append({"location": label, "votes": votes, "sources": sources})

        return {"neighbors": neighbors, "recommendations": recommendations}


class DatasetManager:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def db_path(self, slug: str, model: str = DEFAULT_MODEL) -> Path:
        """Path for sqlite-vec database."""
        safe_model = model.replace("/", "_")
        return self.data_dir / f"{slug}_{safe_model}.db"

    def get_or_create_corpus(self, slug: str, model: str = DEFAULT_MODEL) -> SqliteVecCorpus:
        """Get or create a corpus for the given slug and model."""
        db_file = self.db_path(slug, model)
        corpus = SqliteVecCorpus(db_file)
        corpus._init_schema()
        return corpus

    def crawl(self, owner: str, repo: str, token: str) -> Tuple[Path, int]:
        slug = slugify_repo(owner, repo)

        # Get existing issues from sqlite
        db_file = self.db_path(slug)
        corpus = self.get_or_create_corpus(slug)
        existing_ids = corpus.get_raw_issue_ids()

        # Convert to list format for crawler compatibility
        existing = None
        if existing_ids:
            existing = corpus.load_raw_issues()

        results = crawl_issues_with_diffs(
            owner=owner,
            repo=repo,
            token=token,
            output_path=str(db_file),  # Not used for saving, just for logging
            existing_results=existing,
        )

        # Save to sqlite
        corpus.save_raw_issues(results)
        corpus.close()

        return db_file, len(results)

    async def embed(
        self, owner: str, repo: str, embedder: Embedder, ctx: Optional[Context] = None
    ) -> Dict[str, Any]:
        slug = slugify_repo(owner, repo)
        db_file = self.db_path(slug, embedder.model)

        corpus = self.get_or_create_corpus(slug, embedder.model)

        if corpus.count_raw_issues() == 0:
            raise FileNotFoundError(f"No crawled issues found. Expected {db_file}.")

        # Get already embedded issue IDs
        embedded_ids = corpus.get_embedded_issue_ids()
        total_issues = corpus.count_raw_issues()

        embedded = 0
        processed = 0

        for issue_id, issue in corpus.iter_raw_issues(batch_size=100):
            processed += 1

            # Report progress to MCP client
            if ctx:
                await ctx.report_progress(progress=processed, total=total_issues)

            # Skip already embedded
            if issue_id in embedded_ids:
                continue

            text = sanitize_issue_body(issue)
            if not text:
                continue

            vector = embedder.embed_text(text)
            corpus.save_embedding(issue_id, vector.tolist(), embedder.model)
            embedded += 1

            logger.info(f"Embedded {processed}/{total_issues} (new: {embedded})")

        # Rebuild vector index
        logger.info(f"Building sqlite-vec index for {owner}/{repo}...")
        if ctx:
            await ctx.report_progress(progress=total_issues, total=total_issues)
        indexed_count = corpus.rebuild_vector_index()

        logger.info(f"sqlite-vec index saved to {db_file}")
        corpus.close()

        return {
            "output_path": str(db_file),
            "total_issues": total_issues,
            "embedded_issues": len(embedded_ids) + embedded,
            "indexed_issues": indexed_count,
            "new_embeddings": embedded,
        }

    def load_corpus(self, owner: str, repo: str, model: str, min_word_count: int = 1) -> SqliteVecCorpus:
        slug = slugify_repo(owner, repo)
        db_file = self.db_path(slug, model)

        if not db_file.exists():
            raise FileNotFoundError(
                f"No sqlite-vec database found at {db_file}. Please run embed_repo first."
            )

        logger.info(f"Loading sqlite-vec index from {db_file}")
        corpus = SqliteVecCorpus(db_file)

        # Rebuild index if empty but has raw issues with embeddings
        if corpus.count() == 0 and corpus.count_raw_issues() > 0:
            logger.info("Vector index empty, rebuilding from raw issues...")
            corpus.rebuild_vector_index(min_word_count=min_word_count)

        return corpus


dataset_manager = DatasetManager(DATA_DIR)
embedder = Embedder(DEFAULT_MODEL)
server = FastMCP("bugfs")


@server.tool()
def crawl_repo(
    owner: str,
    repo: str,
    state: str = "closed",
) -> Dict[str, Any]:
    """Crawl GitHub issues plus associated commit diffs for the target repository."""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is required to crawl issues.")
    path, count = dataset_manager.crawl(owner, repo, token)
    return {"issues_path": str(path), "issues_crawled": count, "state": state}


@server.tool()
async def embed_repo(
    owner: str,
    repo: str,
    model: str = "",
    ctx: Context = None,
) -> Dict[str, Any]:
    """Embed every crawled issue (title + body) using the chosen embedding model."""
    if model and model != embedder.model:
        custom_embedder = Embedder(model)
    else:
        custom_embedder = embedder
    result = await dataset_manager.embed(owner, repo, custom_embedder, ctx=ctx)
    return result


@server.tool(name="run_BUGFS")
def run_BUGFS(
    owner: str,
    repo: str,
    title: str,
    body: str,
    neighbors: int = 10,
    max_candidates: int = 10,
    model: str = "",
) -> Dict[str, Any]:
    """
    Recommend bug locations (file:function) by retrieving similar historical issues.
    """
    chosen_model = model if model else embedder.model
    corpus = dataset_manager.load_corpus(owner, repo, chosen_model)
    query_text = re.sub(r"\s+", " ", f"{title.strip()}\n\n{body.strip()}".strip()).strip()
    if not query_text:
        raise ValueError("title+body must contain text.")
    query_embedder = embedder if chosen_model == embedder.model else Embedder(chosen_model)
    query_vector = query_embedder.embed_text(query_text)
    return corpus.recommend(query_vector, neighbor_count=neighbors, max_candidates=max_candidates)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="BUGFS MCP Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    args = parser.parse_args()

    server.run(transport="sse", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
