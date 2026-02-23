# Coordinating agent work on GitHub issues

To analyze open issues, generate prompts, and launch parallel worker agents, follow `agents/coordinator_guide.md`. This uses the GitHub issue tools in `~/git/lab_tools/github/` — see `agents/github_tools_guide.md` for the bitsandbytes-specific reference.

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

# Benchmarking

Benchmark scripts live in `benchmarks/`. The two kbit-specific ones:

- `bench_hadamard.py` — Hadamard rotation kernel + M=1 pipeline (rotation + scalar GEMV) vs cuBLAS FP16. Quick focused benchmark for the decode path.
- `bench_kbit_vlm.py` — Comprehensive sweep across all VLM-relevant M values (1 to 1024), all kernel variants (scalar GEMV, MMA, dequant+cuBLAS), all k values (2-5), with and without Hadamard rotation. Qwen3-Coder-Next 70B shapes.

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
