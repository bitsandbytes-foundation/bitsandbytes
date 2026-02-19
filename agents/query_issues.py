#!/usr/bin/env python3
"""Search and query GitHub issues from the local JSON data file.

Optimized for agent consumption: quality and flexibility first, then compactness.

Examples:
    # List all open issues (one line each)
    python3 github/query_issues.py list
    python3 github/query_issues.py list --state closed --sort comments --limit 20

    # Keyword search across titles and bodies
    python3 github/query_issues.py search "NF4 quantization"
    python3 github/query_issues.py search --label "Bug" --state open "memory"

    # Find issues related to a specific issue
    python3 github/query_issues.py related 1848
    python3 github/query_issues.py related 1848 --state closed -v

    # Find related issues for multiple issues at once
    python3 github/query_issues.py batch-related 1848 1851 1852

    # Show full detail for a specific issue (body + all comments)
    python3 github/query_issues.py show 1848
    python3 github/query_issues.py show --brief 1848

    # Top open issues by reactions
    python3 github/query_issues.py top

    # Summary statistics
    python3 github/query_issues.py stats
"""

import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_DATA = Path(__file__).parent / "bitsandbytes_issues.json"

# Words too common to be useful for matching
STOPWORDS = frozenset({
    # General English stopwords
    'the', 'and', 'for', 'with', 'this', 'that', 'from', 'have', 'has', 'was',
    'are', 'but', 'not', 'you', 'all', 'can', 'had', 'one', 'our', 'out', 'were',
    'been', 'some', 'them', 'than', 'its', 'over', 'will', 'would', 'could',
    'should', 'into', 'also', 'just', 'more', 'when', 'what', 'which', 'their',
    'about', 'there', 'because', 'does', 'like', 'using', 'used', 'use', 'how',
    'please', 'help', 'thank', 'thanks', 'tried', 'trying', 'working', 'getting',
    'running', 'following', 'seems', 'able', 'want', 'need', 'any', 'here', 'then',
    'other', 'being', 'after', 'before', 'only', 'same', 'still', 'make', 'even',
    'most', 'such', 'take', 'come', 'each', 'those', 'very', 'well',
    # Repo-specific: appear in majority of issues, not discriminative
    'bitsandbytes', 'issue', 'error', 'cuda', 'gpu', 'model', 'file', 'work',
    'install', 'pip', 'python', 'import', 'version', 'torch', 'support',
    'available', 'found', 'setup', 'failed', 'library', 'module', 'package',
    'system', 'run', 'load', 'bit', 'get', 'bug', 'report', 'info',
})


def load_data(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def all_issues(data: dict) -> list[dict]:
    return data['open_issues'] + data['closed_issues']


def format_compact(issue: dict) -> str:
    """One-line summary of an issue."""
    labels = ', '.join(issue['labels'][:3]) if issue['labels'] else '-'
    thumbs = issue['reactions'].get('THUMBS_UP', 0)
    return (f"#{issue['number']:<5d} {issue['state']:<6s} "
            f"[{labels}] ({issue['comment_count']}c {thumbs}\u2191) "
            f"{issue['title'][:80]}")


def format_list_line(issue: dict) -> str:
    """Compact one-line summary for list view, with date and key metadata."""
    labels = ', '.join(issue['labels'][:3]) if issue['labels'] else '-'
    thumbs = issue['reactions'].get('THUMBS_UP', 0)
    prs = [t for t in issue['timeline']
           if t['type'] == 'CrossReferencedEvent'
           and t.get('source_type') == 'PullRequest'
           and t.get('source_state') == 'OPEN']
    pr_marker = f" PR#{prs[0]['source_number']}" if prs else ""
    return (f"#{issue['number']:<5d} {issue['updated_at'][:10]} "
            f"[{labels}] {issue['comment_count']}c {thumbs}\u2191"
            f"{pr_marker}  {issue['title'][:70]}")


def format_detail(issue: dict, brief: bool = False) -> str:
    """Full detail view of an issue including body and comments."""
    lines = [
        f"#{issue['number']}: {issue['title']}",
        f"State: {issue['state']}  Author: {issue['author']}  "
        f"Created: {issue['created_at'][:10]}  Updated: {issue['updated_at'][:10]}",
        f"Labels: {', '.join(issue['labels']) or 'none'}",
        f"Assignees: {', '.join(issue['assignees']) or 'none'}",
    ]
    if issue['reactions']:
        rxn = '  '.join(f"{k}:{v}" for k, v in issue['reactions'].items())
        lines.append(f"Reactions: {rxn}")
    lines.append(f"Comments: {issue['comment_count']}")

    # Cross-references (PRs and issues)
    xrefs = [t for t in issue['timeline'] if t['type'] == 'CrossReferencedEvent']
    if xrefs:
        lines.append(f"Cross-references ({len(xrefs)}):")
        for x in xrefs[:15]:
            lines.append(f"  {x['source_type']} #{x['source_number']} "
                         f"[{x['source_state']}]: {x['source_title'][:60]}")

    lines.append("")

    # Body
    body = (issue['body'] or '').strip()
    if brief:
        if len(body) > 1000:
            body = body[:1000] + "\n... [truncated, use show without --brief for full]"
    else:
        # Full body, but cap at 5000 chars for very long issues
        if len(body) > 5000:
            body = body[:5000] + "\n... [truncated at 5000 chars]"
    lines.append(body)

    # Comments
    if issue['comments']:
        lines.append("")
        lines.append(f"--- Comments ({issue['comment_count']}) ---")
        comments = issue['comments']
        if brief:
            # In brief mode, show just first and last comment
            to_show = []
            if comments:
                to_show.append(('first', comments[0]))
            if len(comments) > 1:
                to_show.append(('last', comments[-1]))
            for label, c in to_show:
                rxn = ''
                if c['reactions']:
                    rxn = ' | ' + ' '.join(f"{k}:{v}" for k, v in c['reactions'].items())
                c_body = c['body'].replace('\n', ' ').strip()[:300]
                lines.append(f"  [{label}] @{c['author'] or '?'} ({c['created_at'][:10]}){rxn}:")
                lines.append(f"    {c_body}")
            if len(comments) > 2:
                lines.append(f"  ... {len(comments) - 2} more comments (use show without --brief)")
        else:
            # Full mode: show all comments
            for idx, c in enumerate(comments):
                rxn = ''
                if c['reactions']:
                    rxn = ' | ' + ' '.join(f"{k}:{v}" for k, v in c['reactions'].items())
                lines.append(f"  [{idx+1}] @{c['author'] or '?'} ({c['created_at'][:10]}){rxn}:")
                c_body = c['body'].strip()
                if len(c_body) > 2000:
                    c_body = c_body[:2000] + "\n    ... [comment truncated]"
                # Indent comment body
                for line in c_body.split('\n'):
                    lines.append(f"    {line}")
                lines.append("")

    return '\n'.join(lines)


def tokenize(text: str) -> set[str]:
    """Extract meaningful lowercase tokens."""
    if not text:
        return set()
    text = text.lower()
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'https?://\S+', '', text)
    words = re.findall(r'[a-z][a-z0-9_.]+', text)
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}


def extract_signatures(text: str) -> set[str]:
    """Extract error types, library names, and technical terms.

    These are specific, discriminative terms — not general words like 'cuda'
    which appear in most issues and add noise.
    """
    if not text:
        return set()
    sigs = set()
    # Specific Python error types (but not generic 'error')
    for m in re.finditer(r'(\w+Error|\w+Exception)', text):
        val = m.group(0).lower()
        if val not in ('error', 'exception'):
            sigs.add(val)
    # Library/module paths
    for m in re.finditer(r'(libcudart|libbitsandbytes|torch\.compile|bnb\.\w+|bitsandbytes\.\w+)', text):
        sigs.add(m.group(0).lower())
    # Quantization methods
    for m in re.finditer(r'(nf4|fp4|int8|int4|qlora|lora|gptq|awq)', text, re.I):
        sigs.add(m.group(0).lower())
    # Platforms (excluding 'cuda' — too common to be useful)
    for m in re.finditer(r'(rocm|windows|macos|apple.?silicon|aarch64|arm64|xpu|ascend|gaudi)', text, re.I):
        sigs.add(m.group(0).lower())
    # Specific component/feature terms
    for m in re.finditer(r'(fsdp|deepspeed|triton|matmul|optimizer|quantiz\w+|dequantiz\w+|checkpoint)', text, re.I):
        sigs.add(m.group(0).lower())
    return sigs


def find_related(target: dict, issues: list[dict], state_filter: str | None = None,
                 limit: int = 15) -> list[tuple]:
    """Find issues related to target. Returns list of (score, issue, sig_overlap, token_overlap)."""
    query_text = target['title'] + ' ' + (target['body'] or '')[:1000]
    query_tokens = tokenize(query_text)
    query_sigs = extract_signatures(query_text)
    query_labels = set(target['labels'])

    scored = []
    for issue in issues:
        if issue['number'] == target['number']:
            continue
        if state_filter and issue['state'] != state_filter:
            continue

        body_preview = (issue['body'] or '')[:200]
        issue_text = issue['title'] + ' ' + body_preview
        issue_tokens = tokenize(issue_text)
        issue_sigs = extract_signatures(issue_text)

        sig_overlap = query_sigs & issue_sigs
        token_overlap = query_tokens & issue_tokens
        label_overlap = query_labels & set(issue['labels'])

        score = len(sig_overlap) * 3 + len(token_overlap) + len(label_overlap)
        if score >= 3:
            scored.append((score, issue, sig_overlap, token_overlap))

    scored.sort(key=lambda x: -x[0])
    return scored[:limit]


def format_related_result(score, issue, sig_ol, tok_ol, verbose=False):
    """Format a single related-issue result."""
    lines = []
    lines.append(f"  {format_compact(issue)}")
    matched = list(sig_ol) + list(tok_ol)
    lines.append(f"    score={score}  matched: {', '.join(sorted(matched)[:8])}")
    if verbose:
        body_preview = (issue['body'] or '').replace('\n', ' ').strip()[:300]
        if body_preview:
            lines.append(f"    Body: {body_preview}")
        # Show last comment (often contains resolution or key info)
        if issue['comments']:
            last = issue['comments'][-1]
            last_body = last['body'].replace('\n', ' ').strip()[:200]
            lines.append(f"    Last comment @{last['author'] or '?'} ({last['created_at'][:10]}): {last_body}")
        lines.append("")
    return '\n'.join(lines)


# ---- Commands ----

def cmd_list(args, data):
    """List issues with compact one-line summaries."""
    if args.state:
        if args.state == 'open':
            issues = list(data['open_issues'])
        else:
            issues = list(data['closed_issues'])
    else:
        issues = list(data['open_issues'])

    if args.label:
        label_lower = args.label.lower()
        issues = [i for i in issues if any(label_lower == l.lower() for l in i['labels'])]

    if args.unlabeled:
        issues = [i for i in issues if not i['labels']]

    # Sort
    sort_key = args.sort or 'updated'
    if sort_key == 'updated':
        issues.sort(key=lambda i: i['updated_at'], reverse=True)
    elif sort_key == 'created':
        issues.sort(key=lambda i: i['created_at'], reverse=True)
    elif sort_key == 'comments':
        issues.sort(key=lambda i: i['comment_count'], reverse=True)
    elif sort_key == 'reactions':
        issues.sort(key=lambda i: i['reactions'].get('THUMBS_UP', 0), reverse=True)

    n = args.limit or len(issues)
    for issue in issues[:n]:
        print(format_list_line(issue))
    if n < len(issues):
        print(f"... {len(issues) - n} more (use --limit to show more)")
    print(f"\n({len(issues)} total)", file=sys.stderr)


def cmd_search(args, data):
    """Search issues by keyword."""
    query = args.query.lower()
    query_words = query.split()
    issues = all_issues(data)

    if args.state:
        state = args.state.upper()
        issues = [i for i in issues if i['state'] == state]

    if args.label:
        label_lower = args.label.lower()
        issues = [i for i in issues if any(label_lower == l.lower() for l in i['labels'])]

    results = []
    for issue in issues:
        text = issue['title'].lower()
        if not args.title_only:
            text += ' ' + (issue['body'] or '').lower()[:2000]
        if all(w in text for w in query_words):
            results.append(issue)

    results.sort(key=lambda i: i['reactions'].get('THUMBS_UP', 0), reverse=True)
    n = args.limit or 20
    for issue in results[:n]:
        print(format_compact(issue))
    if len(results) > n:
        print(f"... {len(results) - n} more results (use --limit to show more)")
    elif not results:
        print("No results found.")
    print(f"\n({len(results)} matches)", file=sys.stderr)


def cmd_related(args, data):
    """Find issues related to a given issue number."""
    issues = all_issues(data)
    issue_map = {i['number']: i for i in issues}

    target = issue_map.get(args.number)
    if not target:
        print(f"Issue #{args.number} not found.", file=sys.stderr)
        sys.exit(1)

    state_filter = args.state.upper() if args.state else None
    results = find_related(target, issues, state_filter, args.limit or 15)

    query_sigs = extract_signatures(target['title'] + ' ' + (target['body'] or '')[:1000])
    print(f"Issues related to #{target['number']}: {target['title'][:70]}")
    print(f"  Signatures: {query_sigs or 'none'}")
    print()

    for score, issue, sig_ol, tok_ol in results:
        print(format_related_result(score, issue, sig_ol, tok_ol, verbose=args.verbose))


def cmd_batch_related(args, data):
    """Find related issues for multiple issues at once."""
    issues = all_issues(data)
    issue_map = {i['number']: i for i in issues}

    state_filter = args.state.upper() if args.state else None
    limit_per = args.limit or 5

    for number in args.numbers:
        target = issue_map.get(number)
        if not target:
            print(f"Issue #{number} not found.", file=sys.stderr)
            continue

        results = find_related(target, issues, state_filter, limit_per)
        query_sigs = extract_signatures(target['title'] + ' ' + (target['body'] or '')[:1000])

        print(f"=== #{target['number']}: {target['title'][:65]} ===")
        print(f"  Labels: {', '.join(target['labels']) or 'none'}  "
              f"Signatures: {query_sigs or 'none'}")

        if results:
            for score, issue, sig_ol, tok_ol in results:
                print(format_related_result(score, issue, sig_ol, tok_ol, verbose=args.verbose))
        else:
            print("  No related issues found.")
            print()


def cmd_show(args, data):
    """Show full detail for one or more issues."""
    issues = all_issues(data)
    issue_map = {i['number']: i for i in issues}

    numbers = args.numbers
    for idx, number in enumerate(numbers):
        target = issue_map.get(number)
        if not target:
            print(f"Issue #{number} not found.", file=sys.stderr)
            continue
        if idx > 0:
            print("\n" + "=" * 72 + "\n")
        print(format_detail(target, brief=args.brief))


def cmd_top(args, data):
    """List top issues by reaction count."""
    issues = data['open_issues']
    if args.label:
        label_lower = args.label.lower()
        issues = [i for i in issues if any(label_lower == l.lower() for l in i['labels'])]

    issues = sorted(issues, key=lambda i: i['reactions'].get('THUMBS_UP', 0), reverse=True)
    n = args.limit or 20
    for issue in issues[:n]:
        print(format_compact(issue))


def cmd_stats(args, data):
    """Show summary statistics."""
    from collections import Counter
    print(f"Repository: {data['repository']}")
    print(f"Fetched: {data['fetched_at'][:19]}")
    print(f"Open: {data['open_count']}  Closed: {data['closed_count']}")
    print()

    label_counts = Counter()
    for i in data['open_issues']:
        for l in i['labels']:
            label_counts[l] += 1

    print("Open issue labels:")
    for label, count in label_counts.most_common():
        print(f"  {count:3d}  {label}")

    unlabeled = sum(1 for i in data['open_issues'] if not i['labels'])
    print(f"  {unlabeled:3d}  (unlabeled)")


def main():
    parser = argparse.ArgumentParser(description="Query GitHub issues from local JSON data.")
    parser.add_argument("-d", "--data", default=str(DEFAULT_DATA), help="Path to issues JSON file")
    sub = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = sub.add_parser("list", help="List issues (one line each)")
    p_list.add_argument("--state", choices=["open", "closed"], help="Filter by state (default: open)")
    p_list.add_argument("--label", help="Filter by label name")
    p_list.add_argument("--unlabeled", action="store_true", help="Only show unlabeled issues")
    p_list.add_argument("--sort", choices=["updated", "created", "comments", "reactions"],
                        help="Sort order (default: updated)")
    p_list.add_argument("--limit", type=int, help="Max results")

    # search
    p_search = sub.add_parser("search", help="Keyword search")
    p_search.add_argument("query", help="Search terms")
    p_search.add_argument("--title-only", action="store_true", help="Search title only (default: title + body)")
    p_search.add_argument("--state", choices=["open", "closed"], help="Filter by state")
    p_search.add_argument("--label", help="Filter by label name")
    p_search.add_argument("--limit", type=int, help="Max results (default 20)")

    # related
    p_related = sub.add_parser("related", help="Find related issues")
    p_related.add_argument("number", type=int, help="Issue number to find related issues for")
    p_related.add_argument("--state", choices=["open", "closed"], help="Only show open or closed")
    p_related.add_argument("--limit", type=int, help="Max results (default 15)")
    p_related.add_argument("-v", "--verbose", action="store_true",
                           help="Show body preview and last comment for each result")

    # batch-related
    p_batch = sub.add_parser("batch-related", help="Find related issues for multiple issues at once")
    p_batch.add_argument("numbers", type=int, nargs="+", help="Issue numbers")
    p_batch.add_argument("--state", choices=["open", "closed"], help="Only show open or closed")
    p_batch.add_argument("--limit", type=int, help="Max results per issue (default 5)")
    p_batch.add_argument("-v", "--verbose", action="store_true",
                         help="Show body preview and last comment for each result")

    # show
    p_show = sub.add_parser("show", help="Show full issue detail (body + comments)")
    p_show.add_argument("numbers", type=int, nargs="+", help="Issue number(s)")
    p_show.add_argument("--brief", action="store_true",
                        help="Truncated body, first+last comment only")

    # top
    p_top = sub.add_parser("top", help="Top open issues by reactions")
    p_top.add_argument("--label", help="Filter by label")
    p_top.add_argument("--limit", type=int, help="Max results (default 20)")

    # stats
    sub.add_parser("stats", help="Summary statistics")

    args = parser.parse_args()
    data = load_data(args.data)

    cmds = {
        'list': cmd_list, 'search': cmd_search, 'related': cmd_related,
        'batch-related': cmd_batch_related, 'show': cmd_show,
        'top': cmd_top, 'stats': cmd_stats,
    }
    cmds[args.command](args, data)


if __name__ == "__main__":
    main()
