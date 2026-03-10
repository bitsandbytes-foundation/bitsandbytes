# MANDATORY: Use git worktrees for all branch work

NEVER work on a fix or feature branch inside the main `~/git/bitsandbytes` checkout. Always create a worktree first:

```bash
cd ~/git/bitsandbytes
git worktree add ~/git/bnb-fix-<NUMBER> -b fix/issue-<NUMBER>
cd ~/git/bnb-fix-<NUMBER>
```

This keeps the main checkout clean and allows parallel sessions. If you are already inside a worktree directory, you do not need to create another one.

**Before creating a worktree**, check the worktree registry for existing ones — see the Git Worktrees section in `~/.claude/CLAUDE.md`. Bitsandbytes-specific naming conventions: `agents/worktree_guide.md`. General worktree guide: `~/git/lab_tools/worktree_guide.md`.

# MANDATORY: Check for existing PRs before starting work

Before working on any issue, check whether a PR already exists:

```bash
gh pr list --search "issue-number OR keyword" --state open
```

If a PR exists, review and build on it instead of starting from scratch. Do not create duplicate work.

# MANDATORY: Run linting before every pull request

Before pushing a PR branch, you MUST run the full pre-commit suite. CI will reject PRs that fail any check:

```bash
pre-commit run --all-files
```

This runs ruff, ruff format, typos, trailing-whitespace, clang-format, and all other CI lint hooks. Review and commit any changes it makes. Do NOT run only `ruff check` and `ruff format` — those are just 2 of 10 hooks. Full details: `agents/linting_guide.md`

# Testing: only run relevant tests

Do NOT run the full test suite — it takes 10+ minutes. Instead, run only the tests that cover the code you changed:

```bash
pytest tests/test_relevant_file.py -v --tb=short -k "relevant_test_name"
```

The full suite will be run separately. Best practices, benchmark data, and known architecture-specific issues: `agents/testing_guide.md`

# Benchmarking

Benchmark scripts live in `benchmarks/`. The two kbit-specific ones:

- `bench_hadamard.py` — Hadamard rotation kernel + M=1 pipeline (rotation + scalar GEMV) vs cuBLAS FP16. Quick focused benchmark for the decode path.
- `bench_kbit_vlm.py` — Comprehensive sweep across all VLM-relevant M values (1 to 1024), all kernel variants (scalar GEMV, MMA, dequant+cuBLAS), all k values (2-5), with and without Hadamard rotation. GLM-4.7 shapes (see `spec.md` § Target Model for layer dimensions).

```bash
# Quick M=1 decode benchmark
python benchmarks/bench_hadamard.py

# Full VLM sweep (all M, all k)
python benchmarks/bench_kbit_vlm.py

# Single k value, subset of M
python benchmarks/bench_kbit_vlm.py --k 4 --m 1,4,16,256,1024

# Higher accuracy (more iterations)
python benchmarks/bench_kbit_vlm.py --inner 1000 --outer 30
```

## CUDA graph benchmarking methodology

Single graph replay has a ~14 us timing floor (on RTX 4090) that masks sub-14 us kernel differences. The benchmarks use **batched graph replay**: replay the graph N times within one event-timed region, then divide. This amortizes the per-replay overhead to ~14/N us per iteration.

The `--inner` flag controls N (replays per measurement). Default 500 gives ~0.03 us amortized overhead. Use `--inner 1000` for the highest accuracy when comparing kernels that differ by < 1 us.

`--outer` controls the number of measurements (default 15). The median is reported to reject outliers.

# Agent Dispatch (the "Dispatcher" role)

To triage open GitHub issues, generate prompt files, and launch parallel worker agents, read `agents/dispatch_guide.md`. If told "you're the Dispatcher" or "please read the Dispatch Guide," that's what this refers to. The dispatch workflow uses the GitHub issue tools in `agents/` — see `agents/github_tools_guide.md` for the bitsandbytes-specific reference.

# Issue maintenance and triage

To identify and close stale, duplicate, or resolved issues: `agents/issue_maintenance_guide.md`. Common closeable patterns (old CUDA setup, Windows pre-support, third-party app issues, etc.) are cataloged in `agents/issue_patterns.md`.

# Pull request review

When tasked with reviewing a pull request, you MUST read these guides before starting the review:

1. `agents/pr_review_guide.md` — The complete review workflow (classification, checklists, verdict format, and posting instructions). This is the primary guide; follow its steps sequentially.
2. `agents/architecture_guide.md` — Codebase architecture and patterns
3. `agents/code_standards.md` — Code quality expectations
4. `agents/api_surface.md` — Public API catalog (for detecting breaking changes)
5. `agents/downstream_integrations.md` — How Transformers, PEFT, Accelerate, TGI, and vLLM depend on bitsandbytes (for assessing downstream impact)
6. `agents/security_guide.md` — Trust model and security checklist (especially for external contributor PRs)

For CUDA kernel changes, also read `agents/kbit_gemm_context.md`. The PR review guide references all of these at the appropriate steps.
