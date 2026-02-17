# Git worktrees for parallel Claude Code sessions

## The problem

When you run Claude Code on a repo, the session builds up context about the files, the branch state, and what it has already done. If you switch branches in the same directory, that context becomes stale -- Claude thinks it is looking at one version of the code, but the files on disk have changed underneath it.

Two Claude sessions in the same directory will also step on each other's file writes. There is no locking mechanism. If session A edits `server.py` while session B is also editing `server.py`, one of them will silently overwrite the other.

Git worktrees solve both problems. A worktree is a second (or third, or fourth) checkout of the same repository in a different directory. Each worktree has its own working tree with its own branch checked out, but they all share the same `.git` history and remotes. No cloning, no duplicate disk usage for the object store, no extra credential setup. You use whatever global `claude` installation you already have.

## Quick start

Suppose your repo lives at `~/git/bitsandbytes` and you are on `main`.

Create a worktree for a feature branch:

```bash
git worktree add ../bitsandbytes-feature-auth -b feature/auth
```

This does two things: creates a new branch `feature/auth` (from your current HEAD) and checks it out into `~/git/bitsandbytes-feature-auth`. The directory is created automatically.

If the branch already exists on the remote:

```bash
git fetch origin
git worktree add ../bitsandbytes-feature-auth feature/auth
```

Now open a second terminal, cd into the new directory, and run `claude`. You have two fully independent sessions -- one in `bitsandbytes` on `main`, one in `bitsandbytes-feature-auth` on `feature/auth`. Same git history, same remotes, completely isolated files.

Create as many as you need:

```bash
git worktree add ../bitsandbytes-fix-422 -b fix/issue-422
git worktree add ../bitsandbytes-experiment -b experiment/new-quantizer
```

## Directory layout

After creating a couple of worktrees, your filesystem looks like this:

```
~/git/
├── bitsandbytes/                  # main branch (original clone)
├── bitsandbytes-feature-auth/     # feature/auth branch
├── bitsandbytes-fix-422/          # fix/issue-422 branch
└── bitsandbytes-experiment/       # experiment/new-quantizer branch
```

Each directory is a full working tree. You can run tests, install dependencies, and launch Claude Code independently in each one. They all share the `.git` object store from the original clone, so disk usage stays low.

## Managing worktrees

List all active worktrees:

```bash
git worktree list
```

Output looks like:

```
/home/you/git/bitsandbytes                 abc1234 [main]
/home/you/git/bitsandbytes-feature-auth    def5678 [feature/auth]
/home/you/git/bitsandbytes-fix-422         ghi9012 [fix/issue-422]
```

Remove a worktree when you are done with it:

```bash
git worktree remove ../bitsandbytes-fix-422
```

If you already deleted the directory manually, clean up the stale reference:

```bash
git worktree prune
```

## Setting up each worktree

Each worktree is its own directory, so project dependencies need to be installed separately. After creating a worktree:

**Python projects:**
```bash
cd ../bitsandbytes-feature-auth
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Node projects:**
```bash
cd ../bitsandbytes-feature-auth
npm install
```

This is a one-time cost per worktree. After that, the environment is fully independent.

## Running parallel Claude Code sessions

The workflow is straightforward:

1. Create a worktree per branch (see above).
2. Open a separate terminal for each worktree.
3. Run `claude` in each terminal.
4. Each session gets its own independent context, file state, and history.

Claude Code sessions are directory-scoped. The session picker (`/resume`) groups sessions by repository, including worktrees, so you can tell them apart. Use `/rename` early to give each session a descriptive name like "auth feature" or "quantizer experiment".

**Do not resume the same session in multiple terminals.** Messages get interleaved and the context becomes incoherent. If you need to branch off an existing session, use `claude --fork-session` to create a clean copy.

## Working with different models

You can run different models in different worktrees. This is useful when you want a faster model for straightforward tasks and a stronger model for harder ones:

```bash
# Terminal 1: use Opus for the complex feature
cd ~/git/bitsandbytes-feature-auth
claude --model opus

# Terminal 2: use Sonnet for a simple bugfix
cd ~/git/bitsandbytes-fix-422
claude --model sonnet
```

Each session is independent, so there is no conflict.

## Merging work back

Worktrees share the same git history, so merging is the same as with regular branches:

```bash
# From the main worktree
cd ~/git/bitsandbytes
git merge feature/auth
```

Or push the branch and create a PR on GitHub as usual:

```bash
# From the feature worktree
cd ~/git/bitsandbytes-feature-auth
git push -u origin feature/auth
gh pr create --title "Add auth support"
```

After merging, clean up:

```bash
git worktree remove ../bitsandbytes-feature-auth
git branch -d feature/auth
```

## Rules and constraints

- **One branch per worktree.** You cannot check out the same branch in two worktrees. Git enforces this. If you need two sessions on the same branch, clone the repo instead.
- **Do not delete the main worktree.** The original clone directory is the "main" worktree. You can remove any worktree you created with `git worktree add`, but not the original.
- **Shared refs.** All worktrees see the same branches, tags, and remotes. A `git fetch` in one worktree updates refs for all of them. Commits are visible across worktrees immediately (though the working tree files are not -- each worktree only shows files for its checked-out branch).
- **Stash is shared.** `git stash` entries are visible from all worktrees. This can be confusing. Prefer committing work-in-progress to a branch rather than stashing.

## Cheat sheet

| Task | Command |
|---|---|
| Create worktree + new branch | `git worktree add ../dir -b branch-name` |
| Create worktree for existing branch | `git worktree add ../dir branch-name` |
| List worktrees | `git worktree list` |
| Remove a worktree | `git worktree remove ../dir` |
| Clean up deleted worktrees | `git worktree prune` |
| Rename Claude session | `/rename` inside the session |
| Resume a session | `/resume` inside claude |
| Fork a session | `claude --fork-session` |
