# Issue Maintenance Guide

You are an issue maintenance agent. Your job is to review open GitHub issues for bitsandbytes, identify issues that should be closed, and close them with helpful comments. You are **not** fixing bugs — you are triaging.

## Prerequisites

Refresh the issue data:

```bash
python3 ~/git/lab_tools/github/fetch_issues.py
```

Read `agents/github_tools_guide.md` for the full reference on query tools, and `agents/issue_patterns.md` for known closeable patterns.

## Step 1: Get the Landscape

```bash
# All open issues
python3 ~/git/lab_tools/github/query_issues.py list

# Low-hanging fruit
python3 ~/git/lab_tools/github/query_issues.py list --label "Duplicate"
python3 ~/git/lab_tools/github/query_issues.py list --label "Proposing to Close"
python3 ~/git/lab_tools/github/query_issues.py list --label "Waiting for Info"
python3 ~/git/lab_tools/github/query_issues.py list --label "Question"
python3 ~/git/lab_tools/github/query_issues.py list --label "Likely Not a BNB Issue"
python3 ~/git/lab_tools/github/query_issues.py list --unlabeled
```

## Step 2: Identify Closeable Issues

For each issue, determine if it matches a known pattern from `agents/issue_patterns.md`. The most common categories are:

### Already triaged by maintainers

- **Labeled `Duplicate`** but still open — close with a comment pointing to the canonical issue.
- **Labeled `Proposing to Close`** — review the reason, close if appropriate.
- **Labeled `Waiting for Info`** with no response for 2+ months — close as stale.
- **Labeled `Likely Not a BNB Issue`** — close with a redirect to the correct project.

### Old version issues

Check the bitsandbytes version in the report. Key version boundaries:
- **< 0.43.0**: Old `cuda_setup/main.py` system (replaced). No official Windows support. Fragile CUDA detection.
- **< 0.45.0**: Before improved C library error messaging (PR #1615).

If the issue was clearly caused by old-version behavior that's been fixed, close it.

### Pattern matching

Read the issue body and tracebacks. Compare against the patterns in `agents/issue_patterns.md`:
- Legacy CUDA setup errors
- Windows pre-support issues
- Missing shared library mismatches
- C library load failures showing as `NameError`/`NoneType`
- Third-party app issues
- Transformers version mismatches
- Questions filed as bugs
- FSDP optimizer issues (duplicate of #1633)

### Stale issues

Issues with no activity for 6+ months, no maintainer engagement, and insufficient information to reproduce. Especially:
- No bitsandbytes version specified
- No traceback or only screenshots
- Reporter never responded to requests for info
- Zero comments, zero reactions

## Step 3: Deep-Dive Suspected Duplicates

When two or more issues look related:

```bash
# Full context on both
python3 ~/git/lab_tools/github/query_issues.py show <NUMBER1> <NUMBER2>

# Find more related issues
python3 ~/git/lab_tools/github/query_issues.py related <NUMBER> -v

# Check if already resolved
python3 ~/git/lab_tools/github/query_issues.py related <NUMBER> --state closed -v
```

Before closing a duplicate, verify:
1. The canonical issue is still open (or was resolved with a fix that covers this too).
2. The duplicate doesn't contain unique information that should be preserved — if it does, add a comment on the canonical issue referencing the useful info before closing.

## Step 4: Close with Helpful Comments

Every closed issue should get a comment that:
1. **Explains why** it's being closed (not just "closing as stale").
2. **Points to the fix or canonical issue** if applicable.
3. **Invites reopening** if the problem persists on the latest version.

Use the closing templates from `agents/issue_patterns.md` as a starting point, but tailor them to the specific issue. Mention the actual version the user was on if known, reference the specific fix if one exists.

Close issues using the `gh` CLI:

```bash
gh issue close <NUMBER> --comment "Your comment here"
```

For duplicates, use the `--reason "not planned"` flag:

```bash
gh issue close <NUMBER> --comment "Closing as duplicate of #XXXX." --reason "not planned"
```

## Step 5: Report Results

After a triage session, output a summary:
- How many issues were closed
- Breakdown by category/pattern
- Any issues that are borderline and need human review
- Any new patterns discovered that should be added to `issue_patterns.md`

## Guidelines

- **Be respectful.** People filed these issues because they were stuck. A clear explanation of what went wrong and how to fix it is valuable even on a closed issue.
- **Don't close genuine bugs.** If there's any chance the issue represents a real bug in current code, leave it open. When in doubt, leave it open.
- **Don't close feature requests** unless they're exact duplicates of another open feature request. Feature requests reflect community demand.
- **Preserve information.** If a duplicate issue contains useful reproduction steps, error details, or workarounds not present in the canonical issue, add a comment on the canonical issue with that info before closing the duplicate.
- **Check the version.** The single most important piece of information is the bitsandbytes version. If it's old and the issue area has been reworked, it's likely closeable.
