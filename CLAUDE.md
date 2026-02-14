# Parallel sessions

To work on multiple branches at once, use git worktrees:

```bash
git worktree add ../bitsandbytes-<branch-name> -b <branch-name>
cd ../bitsandbytes-<branch-name>
claude
```

Full guide: `agents/worktree_guide.md`

# Testing

Run the test suite with 4 parallel workers (optimal for any machine):

```bash
pytest tests/ -v --tb=short -n 4
```

Best practices, benchmark data, and known architecture-specific issues: `agents/testing_guide.md`
