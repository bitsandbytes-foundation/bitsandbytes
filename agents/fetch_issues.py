#!/usr/bin/env python3
"""Fetch all issues (open and closed) from a GitHub repository via GraphQL and store as structured JSON."""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

GRAPHQL_QUERY = """
query($owner: String!, $repo: String!, $cursor: String, $states: [IssueState!]) {
  repository(owner: $owner, name: $repo) {
    issues(states: $states, first: 100, after: $cursor, orderBy: {field: CREATED_AT, direction: ASC}) {
      totalCount
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        number
        title
        body
        state
        createdAt
        updatedAt
        closedAt
        author { login }
        assignees(first: 10) { nodes { login } }
        labels(first: 20) { nodes { name } }
        milestone { title number dueOn }
        reactionGroups { content users { totalCount } }
        comments(first: 100) {
          totalCount
          nodes {
            author { login }
            body
            createdAt
            updatedAt
            reactionGroups { content users { totalCount } }
          }
        }
        timelineItems(first: 50, itemTypes: [CROSS_REFERENCED_EVENT, REFERENCED_EVENT, CLOSED_EVENT, REOPENED_EVENT, LABELED_EVENT, UNLABELED_EVENT, CONNECTED_EVENT]) {
          nodes {
            __typename
            ... on CrossReferencedEvent {
              createdAt
              source {
                __typename
                ... on PullRequest { number title state url }
                ... on Issue { number title state url }
              }
            }
            ... on LabeledEvent { label { name } createdAt }
            ... on UnlabeledEvent { label { name } createdAt }
            ... on ClosedEvent { createdAt }
            ... on ReopenedEvent { createdAt }
          }
        }
      }
    }
  }
  rateLimit { cost remaining resetAt }
}
"""


def gh_graphql(query: str, variables: dict) -> dict:
    """Execute a GraphQL query via the gh CLI, passing the full payload as JSON on stdin."""
    clean_vars = {k: v for k, v in variables.items() if v is not None}
    payload = json.dumps({"query": query, "variables": clean_vars})
    result = subprocess.run(
        ["gh", "api", "graphql", "--input", "-"],
        input=payload, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh api graphql failed: {result.stderr}")
    return json.loads(result.stdout)


def transform_reactions(reaction_groups: list) -> dict:
    """Convert reactionGroups to a flat dict, dropping zeros."""
    reactions = {}
    for rg in reaction_groups:
        count = rg["users"]["totalCount"]
        if count > 0:
            reactions[rg["content"]] = count
    return reactions


def transform_timeline_event(event: dict) -> dict | None:
    """Flatten a timeline event node."""
    typename = event.get("__typename")
    if typename == "CrossReferencedEvent":
        source = event.get("source", {})
        return {
            "type": "CrossReferencedEvent",
            "created_at": event.get("createdAt"),
            "source_type": source.get("__typename"),
            "source_number": source.get("number"),
            "source_title": source.get("title"),
            "source_state": source.get("state"),
            "source_url": source.get("url"),
        }
    elif typename in ("LabeledEvent", "UnlabeledEvent"):
        return {
            "type": typename,
            "label": event.get("label", {}).get("name"),
            "created_at": event.get("createdAt"),
        }
    elif typename in ("ClosedEvent", "ReopenedEvent"):
        return {
            "type": typename,
            "created_at": event.get("createdAt"),
        }
    return None


def transform_issue(raw: dict) -> dict:
    """Transform a raw GraphQL issue node into our clean structure."""
    comments = []
    for c in raw["comments"]["nodes"]:
        comments.append({
            "author": c["author"]["login"] if c.get("author") else None,
            "body": c["body"],
            "created_at": c["createdAt"],
            "updated_at": c["updatedAt"],
            "reactions": transform_reactions(c.get("reactionGroups", [])),
        })

    timeline = []
    for t in raw["timelineItems"]["nodes"]:
        transformed = transform_timeline_event(t)
        if transformed:
            timeline.append(transformed)

    return {
        "number": raw["number"],
        "title": raw["title"],
        "body": raw["body"],
        "state": raw["state"],
        "author": raw["author"]["login"] if raw.get("author") else None,
        "created_at": raw["createdAt"],
        "updated_at": raw["updatedAt"],
        "closed_at": raw["closedAt"],
        "assignees": [a["login"] for a in raw["assignees"]["nodes"]],
        "labels": [l["name"] for l in raw["labels"]["nodes"]],
        "milestone": raw.get("milestone"),
        "reactions": transform_reactions(raw.get("reactionGroups", [])),
        "comment_count": raw["comments"]["totalCount"],
        "comments": comments,
        "timeline": timeline,
    }


def fetch_all_issues(owner: str, repo: str, states: list[str] | None = None) -> list[dict]:
    """Fetch issues with pagination and exponential backoff."""
    if states is None:
        states = ["OPEN"]
    all_issues = []
    cursor = None
    page = 1
    max_retries = 5
    label = "/".join(s.lower() for s in states)

    while True:
        for attempt in range(max_retries):
            try:
                print(f"Fetching {label} issues page {page}...", file=sys.stderr)
                data = gh_graphql(GRAPHQL_QUERY, {
                    "owner": owner, "repo": repo, "cursor": cursor, "states": states,
                })
                break
            except RuntimeError as e:
                wait = min(2 ** attempt, 60)
                print(f"Error on attempt {attempt + 1}: {e}", file=sys.stderr)
                if attempt < max_retries - 1:
                    print(f"Retrying in {wait}s...", file=sys.stderr)
                    time.sleep(wait)
                else:
                    raise

        rate = data["data"]["rateLimit"]
        print(f"  Rate limit: {rate['remaining']} remaining, cost: {rate['cost']}", file=sys.stderr)

        if rate["remaining"] < 100:
            reset_at = datetime.fromisoformat(rate["resetAt"].replace("Z", "+00:00"))
            wait_seconds = (reset_at - datetime.now(timezone.utc)).total_seconds() + 5
            if wait_seconds > 0:
                print(f"  Rate limit low, waiting {wait_seconds:.0f}s until reset...", file=sys.stderr)
                time.sleep(wait_seconds)

        issues_data = data["data"]["repository"]["issues"]
        raw_issues = issues_data["nodes"]
        total = issues_data["totalCount"]

        for raw in raw_issues:
            all_issues.append(transform_issue(raw))

        print(f"  Fetched {len(all_issues)}/{total} issues", file=sys.stderr)

        page_info = issues_data["pageInfo"]
        if not page_info["hasNextPage"]:
            break

        cursor = page_info["endCursor"]
        page += 1

    return all_issues


def main():
    parser = argparse.ArgumentParser(description="Fetch all GitHub issues into a JSON file.")
    parser.add_argument("--owner", default="bitsandbytes-foundation", help="Repository owner")
    parser.add_argument("--repo", default="bitsandbytes", help="Repository name")
    parser.add_argument("--open-only", action="store_true", help="Only fetch open issues")
    parser.add_argument("-o", "--output", default=None,
                        help="Output JSON file path (default: <repo>_issues.json in script dir)")
    args = parser.parse_args()

    output_path = args.output or str(Path(__file__).parent / f"{args.repo}_issues.json")

    open_issues = fetch_all_issues(args.owner, args.repo, ["OPEN"])
    print(file=sys.stderr)

    if args.open_only:
        closed_issues = []
    else:
        closed_issues = fetch_all_issues(args.owner, args.repo, ["CLOSED"])
        print(file=sys.stderr)

    result = {
        "repository": f"{args.owner}/{args.repo}",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "open_issues": open_issues,
        "open_count": len(open_issues),
        "closed_issues": closed_issues,
        "closed_count": len(closed_issues),
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(open_issues)} open + {len(closed_issues)} closed issues to {output_path}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
