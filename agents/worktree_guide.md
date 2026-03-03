# Worktree conventions for bitsandbytes

For general worktree concepts, setup, and the worktree registry, see `~/git/lab_tools/worktree_guide.md`. This file covers bitsandbytes-specific conventions only.

## Naming

Worktree directories for bitsandbytes use the short prefix `bnb-`:

| Purpose | Directory | Branch |
|---|---|---|
| Issue fix | `~/git/bnb-fix-<NUMBER>` | `fix/issue-<NUMBER>` |
| Feature | `~/git/bitsandbytes-<name>` | `feature/<name>` |
| Experiment | `~/git/bnb-kbit-gemm` | `feature/kbit-gemv-v8` |
| Deprecation | `~/git/bnb-deprecation` | `deprecation` |

For issue-related work, always include the issue number. The dispatch workflow generates worktrees with this pattern automatically.

## Quick start

```bash
cd ~/git/bitsandbytes
git worktree add ~/git/bnb-fix-<NUMBER> -b fix/issue-<NUMBER>
cd ~/git/bnb-fix-<NUMBER>
```

## Build and test

After creating a worktree, read `agents/testing_guide.md` for build instructions. Run only the tests relevant to your change — not the full suite.

## Dispatch workflow

When launched via the dispatch guide (`agents/dispatch_guide.md`), worker agents receive prompt files that include worktree creation commands. The prompts follow the naming conventions above. Workers must create their worktree before starting work.

## Completion

After implementing and verifying a fix:

1. Commit with a message referencing the issue: `git commit -m "Fix <description> (#<NUMBER>)"`
2. Push: `git push -u origin fix/issue-<NUMBER>`
3. Create a PR with `gh pr create` — include "Fixes #<NUMBER>" in the body.
4. The worktree manager cron job will clean up the worktree after the PR is merged.
