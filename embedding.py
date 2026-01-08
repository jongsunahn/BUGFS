import json
from openai import OpenAI
from dotenv import load_dotenv
import os
import time
import numpy as np

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

INPUT_JSON = "spinnaker_issues_.json"
OUTPUT_JSON = "spinnaker_embeddings_.json"
MODEL = "text-embedding-3-small"

CHUNK_CHAR_LIMIT = 3000   # 필요하면 조절


def load_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] failed to load existing {path}: {e}")
        return {}


def atomic_save(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)


def load_issues(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_issue_text(issue):
    """title + body 묶어서 임베딩 입력 텍스트 생성"""
    title = issue.get("title", "")
    body = issue.get("body", "")
    return f"{title}\n\n{body}".strip()


def split_into_chunks(text: str, limit: int = CHUNK_CHAR_LIMIT):
    """긴 텍스트를 limit 길이만큼 청크로 나눔."""
    return [text[i:i + limit] for i in range(0, len(text), limit)]


def embed_text_chunks(client, chunks):
    """여러 chunk를 embedding 후 평균 벡터 반환"""
    res = client.embeddings.create(
        model=MODEL,
        input=chunks
    )

    vectors = [np.array(item.embedding) for item in res.data]
    avg_vec = np.mean(vectors, axis=0)
    return avg_vec.tolist()


def format_seconds(sec: float) -> str:
    """초를 'HH:MM:SS' 문자열로 포맷"""
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    client = OpenAI()
    issues = load_issues(INPUT_JSON)

    # 🔥 기존 결과 로드 후 이어서 진행
    embeddings = load_json(OUTPUT_JSON)
    total_issues = len(issues)
    already_embedded = len(embeddings)

    print(f"[INFO] 전체 이슈 수: {total_issues}")
    print(f"[INFO] 이미 임베딩된 이슈 수: {already_embedded}")
    print(f"[INFO] 남은 이슈 수: {total_issues - already_embedded}")

    start_time = time.time()
    newly_processed = 0  # 이번 실행에서 새로 임베딩한 개수

    for idx, issue in enumerate(issues, start=1):
        issue_id = str(issue["id"])

        # 🔥 이미 처리한 issue는 스킵
        if issue_id in embeddings:
            # 진행도 표시용: 스킵도 '이미 완료된 것'으로 취급
            progress_count = already_embedded + newly_processed
            progress_percent = (progress_count / total_issues) * 100 if total_issues > 0 else 0.0
            print(f"[SKIP] issue {issue_id} already embedded "
                  f"({progress_count}/{total_issues}, {progress_percent:.2f}%)")
            continue

        text = build_issue_text(issue)
        if not text.strip():
            print(f"[WARN] issue {issue_id} has empty text. skipping")
            continue

        progress_count = already_embedded + newly_processed
        progress_percent = (progress_count / total_issues) * 100 if total_issues > 0 else 0.0

        print(f"[INFO] embedding issue {issue_id}, index={idx}/{total_issues}, "
              f"진행도={progress_percent:.2f}%, text length={len(text)}")

        chunks = split_into_chunks(text)
        print(f"[INFO] → split into {len(chunks)} chunks")

        # 🔥 API 재시도 로직
        for attempt in range(5):
            try:
                t0 = time.time()
                vec = embed_text_chunks(client, chunks)
                t1 = time.time()
                elapsed_for_issue = t1 - t0
                break
            except Exception as e:
                print(f"[ERROR] embedding issue {issue_id}, attempt {attempt+1}/5: {e}")
                time.sleep(2)
        else:
            print(f"[FAIL] issue {issue_id} skipped after 5 retries")
            continue

        embeddings[issue_id] = vec
        newly_processed += 1

        # ✅ ETA 계산
        total_elapsed = time.time() - start_time
        if newly_processed > 0:
            avg_per_issue = total_elapsed / newly_processed
            remaining_issues = total_issues - (already_embedded + newly_processed)
            eta_seconds = max(0, remaining_issues * avg_per_issue)
            eta_str = format_seconds(eta_seconds)
        else:
            eta_str = "N/A"

        print(f"[OK] Embedded issue {issue_id}, vector length={len(vec)}, "
              f"소요시간(방금): {elapsed_for_issue:.2f}s, 예상 남은 시간: {eta_str}")

        # 🔥 즉시 저장
        atomic_save(OUTPUT_JSON, embeddings)
        print(f"[SAVE] 현재까지 {already_embedded + newly_processed}/{total_issues}개 저장 완료 → {OUTPUT_JSON}")

    print(f"[DONE] total embedded issues: {len(embeddings)}")
    atomic_save(OUTPUT_JSON, embeddings)
    print("[FINAL SAVE] completed")


if __name__ == "__main__":
    main()