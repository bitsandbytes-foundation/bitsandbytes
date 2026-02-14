# MANDATORY: Use git worktrees for all branch work

NEVER work on a fix or feature branch inside the main `~/git/bitsandbytes` checkout. Always create a worktree first:

```bash
cd ~/git/bitsandbytes
git worktree add ~/git/bnb-fix-<NUMBER> -b fix/issue-<NUMBER>
cd ~/git/bnb-fix-<NUMBER>
```

This keeps the main checkout clean and allows parallel sessions. If you are already inside a worktree directory, you do not need to create another one. Full guide: `agents/worktree_guide.md`

# MANDATORY: Check for existing PRs before starting work

Before working on any issue, check whether a PR already exists:

```bash
gh pr list --search "issue-number OR keyword" --state open
```

If a PR exists, review and build on it instead of starting from scratch. Do not create duplicate work.

# Testing: only run relevant tests

Do NOT run the full test suite — it takes 10+ minutes. Instead, run only the tests that cover the code you changed:

```bash
pytest tests/test_relevant_file.py -v --tb=short -k "relevant_test_name"
```

The full suite will be run separately. Best practices and known issues: `agents/testing_guide.md`

# Coordinating agent work on GitHub issues

To analyze open issues, generate prompts, and launch parallel worker agents, follow `agents/coordinator_guide.md`. This uses the GitHub issue tools in `~/git/lab_tools/github/` — see `agents/github_tools_guide.md` for the bitsandbytes-specific reference.

# Issue maintenance and triage

To identify and close stale, duplicate, or resolved issues: `agents/issue_maintenance_guide.md`. Common closeable patterns (old CUDA setup, Windows pre-support, third-party app issues, etc.) are cataloged in `agents/issue_patterns.md`.
