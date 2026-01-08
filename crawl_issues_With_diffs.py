#!/usr/bin/env python3
"""
GitHub Issue + Commit Diff Crawler

- flask_issues.json 과 100% 동일한 구조로 JSON 생성
  [
    {
      "id": <issue_id>,
      "url": "<issue_api_url>",
      "title": "<issue_title>",
      "body": "<issue_body>",
      "commits": [
        {
          "filename": "<file_path>",
          "patch": "<unified_diff_patch>",
          "date": "<ISO8601_commit_date>"
        },
        ...
      ]
    },
    ...
  ]

사용 예:
    export GITHUB_TOKEN=xxxx
    python crawl_issues_with_diffs.py pallets flask flask_issues.json
"""

import os
import sys
import time
import json
import requests
from typing import List, Dict, Any, Set, Optional
import dotenv
from tqdm import tqdm

dotenv.load_dotenv()
os.environ["GITHUB_TOKEN"] = os.getenv("GITHUB_TOKEN", "")

GITHUB_API = "https://api.github.com"


def make_session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
        }
    )
    return s


def fetch_issues(
    session: requests.Session,
    owner: str,
    repo: str,
    state: str = "all",
    per_page: int = 100,
) -> List[Dict[str, Any]]:
    """모든 이슈(이슈 + PR)를 페이징 하면서 가져옴."""
    issues: List[Dict[str, Any]] = []
    page = 1

    pbar = tqdm(desc=f"Fetching issues from {owner}/{repo}", unit="page")

    while True:
        url = f"{GITHUB_API}/repos/{owner}/{repo}/issues"
        params = {
            "state": state,
            "per_page": per_page,
            "page": page,
        }
        try:
            resp = session.get(url, params=params)
        except requests.RequestException as e:
            tqdm.write(f"[ERROR] fetch_issues page={page} exception: {e}")
            break

        if resp.status_code != 200:
            tqdm.write(f"[WARN] fetch_issues page={page} status={resp.status_code}")
            break

        batch = resp.json()
        if not batch:
            break

        issues.extend(batch)
        pbar.update(1)
        pbar.set_postfix({"total_issues": len(issues)})
        page += 1

        time.sleep(0.2)

    pbar.close()
    return issues


def fetch_issue_timeline(
    session: requests.Session,
    owner: str,
    repo: str,
    issue_number: int,
) -> List[Dict[str, Any]]:
    """
    Issue timeline API 로 cross-referenced PR 등을 찾음.
    """
    url = f"{GITHUB_API}/repos/{owner}/{repo}/issues/{issue_number}/timeline"

    headers = {
        "Accept": "application/vnd.github.mockingbird-preview+json"
    }
    try:
        resp = session.get(url, headers=headers)
    except requests.RequestException as e:
        print(f"[WARN] fetch_issue_timeline #{issue_number} exception: {e}")
        return []

    if resp.status_code != 200:
        print(f"[WARN] fetch_issue_timeline #{issue_number} status={resp.status_code}")
        return []

    events = resp.json()
    time.sleep(0.2)
    return events


def collect_pr_numbers_for_issue(
    session: requests.Session,
    owner: str,
    repo: str,
    issue: Dict[str, Any],
) -> Set[int]:
    """
    한 issue 에 대해 연관된 PR 번호들을 수집:
      1) 이슈 자체가 PR인 경우
      2) timeline cross-referenced 로 연결된 PR들
    """
    pr_numbers: Set[int] = set()
    issue_number = issue["number"]

    # 1) 이슈가 곧 PR인 경우
    if "pull_request" in issue:
        pr_numbers.add(issue_number)

    # 2) 타임라인에서 cross-referenced PR 찾기
    events = fetch_issue_timeline(session, owner, repo, issue_number)
    for ev in events:
        if ev.get("event") == "cross-referenced":
            src_issue = ev.get("source", {}).get("issue", {})
            if src_issue.get("pull_request"):
                pr_num = src_issue.get("number")
                if isinstance(pr_num, int):
                    pr_numbers.add(pr_num)

    return pr_numbers


def fetch_pr_commits(
    session: requests.Session,
    owner: str,
    repo: str,
    pr_number: int,
) -> List[Dict[str, Any]]:
    """PR에 포함된 commit 리스트 가져오기."""
    url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}/commits"
    try:
        resp = session.get(url)
    except requests.RequestException as e:
        print(f"[WARN] fetch_pr_commits PR#{pr_number} exception: {e}")
        return []

    if resp.status_code != 200:
        print(f"[WARN] fetch_pr_commits PR#{pr_number} status={resp.status_code}")
        return []

    commits = resp.json()
    time.sleep(0.2)
    return commits


def fetch_commit_files(
    session: requests.Session,
    owner: str,
    repo: str,
    sha: str,
) -> Optional[Dict[str, Any]]:
    """commit 상세(파일 + patch 포함) 가져오기."""
    url = f"{GITHUB_API}/repos/{owner}/{repo}/commits/{sha}"
    try:
        resp = session.get(url)
    except requests.RequestException as e:
        print(f"[WARN] fetch_commit_files sha={sha} exception: {e}")
        return None

    if resp.status_code != 200:
        print(f"[WARN] fetch_commit_files sha={sha} status={resp.status_code}")
        return None

    commit_detail = resp.json()
    time.sleep(0.2)
    return commit_detail


def atomic_save_json(path: str, data: Any) -> None:
    """임시 파일에 저장 후 rename 으로 교체 (중간에 죽어도 파일 깨지는 것 방지)."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp_path, path)


def crawl_issues_with_diffs(
    owner: str,
    repo: str,
    token: str,
    output_path: str,
    existing_results: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    - 기존에 저장된 결과가 있으면 이어서 크롤링.
    - 각 issue 처리 후마다 output_path 에 바로 저장.
    """
    session = make_session(token)
    all_issues = fetch_issues(session, owner, repo)

    # 이미 저장된 결과가 있으면 이어서
    if existing_results is None:
        results: List[Dict[str, Any]] = []
    else:
        results = list(existing_results)

    processed_ids: Set[int] = {item["id"] for item in results}

    # Filter issues that need processing
    issues_to_process = [i for i in all_issues if i["id"] not in processed_ids]

    pbar = tqdm(
        issues_to_process,
        desc=f"Crawling {owner}/{repo}",
        unit="issue",
        initial=len(processed_ids),
        total=len(all_issues),
    )

    for issue in pbar:
        issue_id = issue["id"]
        issue_url = issue["url"]
        title = issue.get("title", "")
        body = issue.get("body", "")

        pbar.set_postfix({"issue": issue["number"]})

        pr_numbers = collect_pr_numbers_for_issue(session, owner, repo, issue)

        commits_entries: List[Dict[str, Any]] = []

        for pr_num in pr_numbers:
            pr_commits = fetch_pr_commits(session, owner, repo, pr_num)

            for c in pr_commits:
                sha = c["sha"]
                commit_detail = fetch_commit_files(session, owner, repo, sha)
                if not commit_detail:
                    continue

                commit_date = (
                    commit_detail
                    .get("commit", {})
                    .get("author", {})
                    .get("date")
                )

                for f in commit_detail.get("files", []):
                    patch = f.get("patch")
                    if not patch:
                        continue

                    filename = f.get("filename")
                    commits_entries.append(
                        {
                            "filename": filename,
                            "patch": patch,
                            "date": commit_date,
                        }
                    )

        issue_record = {
            "id": issue_id,
            "url": issue_url,
            "title": title,
            "body": body,
            "commits": commits_entries,
        }

        results.append(issue_record)
        processed_ids.add(issue_id)

        atomic_save_json(output_path, results)

    pbar.close()

    return results


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: python crawl_issues_with_diffs.py <owner> <repo> <output_json>\n"
            "Example:\n"
            '  python crawl_issues_with_diffs.py pallets flask flask_issues.json'
        )
        sys.exit(1)

    owner = sys.argv[1]
    repo = sys.argv[2]
    output_path = sys.argv[3]

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: GITHUB_TOKEN 환경변수를 설정해 주세요.")
        sys.exit(1)

    # 🔹 기존 결과가 있으면 먼저 로드해서 이어서 크롤링
    existing_results: Optional[List[Dict[str, Any]]] = None
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
            print(f"[INFO] loaded existing results from {output_path}")
        except Exception as e:
            print(f"[WARN] failed to load existing {output_path}: {e}")
            existing_results = None

    results = crawl_issues_with_diffs(owner, repo, token, output_path, existing_results)

    # 마지막으로 한 번 더 저장 (안전)
    atomic_save_json(output_path, results)
    print(f"[DONE] saved {len(results)} issues to {output_path}")


if __name__ == "__main__":
    main()