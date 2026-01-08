#!/usr/bin/env python3
"""
Migration script: Convert existing JSON embedding files to FAISS format.
"""

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

DATA_DIR = Path("data")
EMBEDDING_DIM = 1536

# Regex patterns from BUGFS_mcp_server.py
HUNK_HEADER_REGEX = re.compile(
    r"@@ -\d+,\d+ \+(?P<start>\d+),(?P<length>\d+) @@(?:\s+(?P<context>.+))?$",
    flags=re.MULTILINE,
)
PYTHON_FUNC_PATTERN = re.compile(r"def\s+(?P<name>\w+)")
JAVA_METHOD_PATTERN = re.compile(
    r"(?:public|private|protected|static|final|abstract|synchronized|void|\w+)\s+(?P<name>\w+)\s*\("
)
CLASS_PATTERN = re.compile(r"(?:class|interface|enum)\s+(?P<name>\w+)")


@dataclass(frozen=True)
class IssueLocation:
    start: int
    length: int
    function: str
    file: str

    @property
    def label(self) -> str:
        return f"{self.file}:{self.function}"


def normalize_filename(path_str: str) -> str:
    path_str = path_str.replace("\\", "/")
    parts = [p for p in path_str.split("/") if p]
    if not parts:
        return path_str
    return parts[-1]


def extract_method_from_context(context: str) -> Optional[str]:
    if not context:
        return None
    match = PYTHON_FUNC_PATTERN.search(context)
    if match:
        return match.group("name")
    match = JAVA_METHOD_PATTERN.search(context)
    if match:
        name = match.group("name")
        if name not in ("if", "for", "while", "switch", "catch", "synchronized", "return", "new", "throw"):
            return name
    match = CLASS_PATTERN.search(context)
    if match:
        return match.group("name")
    return None


def extract_locations(issue: Dict[str, Any]) -> List[IssueLocation]:
    locations: List[IssueLocation] = []
    seen: set[tuple[str, str]] = set()
    for commit in issue.get("commits", []):
        filename = normalize_filename(commit.get("filename", "unknown"))
        patch = commit.get("patch")
        if not patch:
            continue
        for match in HUNK_HEADER_REGEX.finditer(patch):
            start = int(match.group("start"))
            length = int(match.group("length"))
            context = match.group("context") or ""
            method_name = extract_method_from_context(context)
            if method_name:
                key = (filename, method_name)
                if key not in seen:
                    seen.add(key)
                    locations.append(
                        IssueLocation(start=start, length=length, function=method_name, file=filename)
                    )
    return locations


def build_issue_text(issue: Dict[str, Any]) -> str:
    title = (issue.get("title") or "").strip()
    body = (issue.get("body") or "").strip()
    return f"{title}\n\n{body}".strip()


def sanitize_issue_body(issue: Dict[str, Any], min_word_count: int = 1) -> Optional[str]:
    text = build_issue_text(issue)
    text = re.sub(r"\s+", " ", text)
    if len(text.split()) < min_word_count:
        return None
    return text


def migrate_json_to_faiss(json_path: Path) -> None:
    """Convert a JSON embedding file to FAISS format."""
    print(f"\n{'='*60}")
    print(f"Migrating: {json_path.name}")
    print(f"File size: {json_path.stat().st_size / (1024*1024):.1f} MB")
    print(f"{'='*60}")

    # Parse filename to get slug and model
    # Format: {slug}_issues_with_{model}.json
    name = json_path.stem  # remove .json
    if "_issues_with_" not in name:
        print(f"⚠️  Skipping {json_path.name} - not an embedding file")
        return

    parts = name.split("_issues_with_")
    slug = parts[0]
    model = parts[1]

    print(f"Slug: {slug}")
    print(f"Model: {model}")

    # Load JSON
    print("Loading JSON...")
    with json_path.open("r", encoding="utf-8") as f:
        issues = json.load(f)
    print(f"Loaded {len(issues)} issues")

    # Build records and embeddings
    print("Processing issues...")
    records: List[Dict[str, Any]] = []
    embeddings: List[np.ndarray] = []
    embeddings_cache: Dict[str, Dict[str, Any]] = {}

    skipped_no_embedding = 0
    skipped_no_locations = 0
    skipped_no_text = 0

    for issue in issues:
        embedding = issue.get("embedding")
        if not embedding:
            skipped_no_embedding += 1
            continue

        # Cache all embeddings (even without locations)
        issue_id = str(issue["id"])
        embeddings_cache[issue_id] = issue

        locations = extract_locations(issue)
        if not locations:
            skipped_no_locations += 1
            continue

        text = sanitize_issue_body(issue)
        if not text:
            skipped_no_text += 1
            continue

        # Convert IssueLocation objects to dicts for pickle compatibility
        locations_as_dicts = [
            {"start": loc.start, "length": loc.length, "function": loc.function, "file": loc.file}
            for loc in locations
        ]
        records.append({
            "id": issue["id"],
            "locations": locations_as_dicts,
            "content": text,
        })
        embeddings.append(np.asarray(embedding, dtype=np.float32))

    print(f"  - With embeddings: {len(embeddings_cache)}")
    print(f"  - Skipped (no embedding): {skipped_no_embedding}")
    print(f"  - Skipped (no locations): {skipped_no_locations}")
    print(f"  - Skipped (no text): {skipped_no_text}")
    print(f"  - Valid for index: {len(records)}")

    if not records:
        print("⚠️  No valid records to index!")
        return

    # Build FAISS index
    print("Building FAISS index...")
    matrix = np.vstack(embeddings).astype(np.float32)
    faiss.normalize_L2(matrix)

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    print(f"Index created with {index.ntotal} vectors, dim={index.d}")

    # Save files
    faiss_path = DATA_DIR / f"{slug}_{model}.faiss"
    metadata_path = DATA_DIR / f"{slug}_{model}_metadata.pkl"
    embeddings_path = DATA_DIR / f"{slug}_{model}_embeddings.pkl"

    print(f"Saving FAISS index to {faiss_path.name}...")
    faiss.write_index(index, str(faiss_path))

    print(f"Saving metadata to {metadata_path.name}...")
    with metadata_path.open("wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saving embeddings cache to {embeddings_path.name}...")
    with embeddings_path.open("wb") as f:
        pickle.dump(embeddings_cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Print size comparison
    old_size = json_path.stat().st_size
    new_size = faiss_path.stat().st_size + metadata_path.stat().st_size + embeddings_path.stat().st_size

    print(f"\n📊 Size comparison:")
    print(f"  - Original JSON: {old_size / (1024*1024):.1f} MB")
    print(f"  - FAISS index:   {faiss_path.stat().st_size / (1024*1024):.1f} MB")
    print(f"  - Metadata:      {metadata_path.stat().st_size / (1024*1024):.1f} MB")
    print(f"  - Embeddings:    {embeddings_path.stat().st_size / (1024*1024):.1f} MB")
    print(f"  - Total new:     {new_size / (1024*1024):.1f} MB")
    print(f"  - Reduction:     {(1 - new_size/old_size)*100:.1f}%")

    print(f"\n✅ Migration complete for {slug}")


def main():
    print("🚀 FAISS Migration Tool")
    print(f"Data directory: {DATA_DIR.absolute()}")

    # Find all embedding JSON files
    json_files = list(DATA_DIR.glob("*_issues_with_*.json"))

    if not json_files:
        print("No embedding files found to migrate!")
        return

    print(f"\nFound {len(json_files)} embedding file(s) to migrate:")
    for f in json_files:
        print(f"  - {f.name} ({f.stat().st_size / (1024*1024):.1f} MB)")

    for json_path in json_files:
        migrate_json_to_faiss(json_path)

    print("\n" + "="*60)
    print("🎉 All migrations complete!")
    print("="*60)


if __name__ == "__main__":
    main()
