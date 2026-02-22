"""Qwen3-Coder-Next 70B weight matmul summary.

Reads benchmark results from .bench_results/ and produces one table per M
value. Each row is a (shape, k) combination. Columns show all kernel timings
side by side, the best kernel, and speedup vs cuBLAS fp16.

Dense shapes have MMA, Scalar, fp16 columns.
MoE shapes have Grouped (scalar), Grp MMA, fp16 (bmm) columns.
"""

import os
import sys


def parse_results(path):
    """Parse a benchmark result file into {(shape, k, M): avg_us}."""
    results = {}
    if not os.path.exists(path):
        return results
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("shape") or line.startswith("---"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    results[(parts[0], int(parts[1]), int(parts[2]))] = float(parts[3])
                except (ValueError, IndexError):
                    continue
    return results


def parse_cublas(path):
    """Parse cuBLAS results. Dense lines have 3 columns, MoE lines have 4."""
    dense = {}
    moe = {}
    if not os.path.exists(path):
        return dense, moe
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("shape") or line.startswith("---"):
                continue
            parts = line.split()
            try:
                if len(parts) == 3:
                    dense[(parts[0], int(parts[1]))] = float(parts[2])
                elif len(parts) == 4:
                    moe[(parts[0], int(parts[1]))] = float(parts[3])
            except (ValueError, IndexError):
                continue
    return dense, moe


def fmt(val):
    """Format a float as right-aligned string, or '-' if None."""
    if val is None:
        return "  -  "
    return f"{val:5.1f}"


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else ".bench_results"

    mma = parse_results(os.path.join(results_dir, "mma.txt"))
    scalar = parse_results(os.path.join(results_dir, "scalar.txt"))
    grouped = parse_results(os.path.join(results_dir, "grouped.txt"))
    grouped_mma = parse_results(os.path.join(results_dir, "grouped_mma.txt"))
    cublas_dense, cublas_moe = parse_cublas(os.path.join(results_dir, "cublas.txt"))

    if not mma and not scalar and not grouped and not grouped_mma:
        print("No benchmark results found. Run bench_ncu.sh first.")
        return

    # All shapes in display order
    dense_shapes = ["gateup", "down", "Q", "O", "KV"]
    moe_shapes = ["moe_gu", "moe_dn"]
    all_shapes = dense_shapes + moe_shapes
    k_bits = [2, 3, 4, 5]

    # Collect all M values
    all_M = set()
    for d in [mma, scalar, grouped, grouped_mma]:
        for key in d:
            all_M.add(key[2])
    all_M = sorted(all_M)

    # Column widths — 6 kernel columns
    SEP = "+"
    HDR = f"{SEP}--------+-----+-------+--------+---------+---------+-------+--------+---------{SEP}"
    TOP = f"{SEP}========+=====+=======+========+=========+=========+=======+========+========={SEP}"

    for M in all_M:
        print(f"\n  M={M}:")
        print(f"  {TOP}")
        print(
            f"  | {'shape':<6} | {'k':>3} | {'MMA':>5} | {'Scalar':>6} | {'Grouped':>7} | {'Grp MMA':>7} | {'fp16':>5} | {'Best':>6} | {'vs fp16':>7} |"
        )
        print(f"  {HDR}")

        for shape in all_shapes:
            is_moe = shape in moe_shapes

            for k in k_bits:
                # Gather timings
                m_us = mma.get((shape, k, M)) if not is_moe else None
                s_us = scalar.get((shape, k, M)) if not is_moe else None
                g_us = grouped.get((shape, k, M)) if is_moe else None
                gm_us = grouped_mma.get((shape, k, M)) if is_moe else None
                fp16 = cublas_moe.get((shape, M)) if is_moe else cublas_dense.get((shape, M))

                # Find best kbit kernel
                candidates = {}
                if m_us is not None:
                    candidates["MMA"] = m_us
                if s_us is not None:
                    candidates["Scalar"] = s_us
                if g_us is not None:
                    candidates["Grouped"] = g_us
                if gm_us is not None:
                    candidates["Grp MMA"] = gm_us

                if candidates:
                    best_name = min(candidates, key=candidates.get)
                    best_us = candidates[best_name]
                else:
                    best_name, best_us = None, None

                if best_us is not None and fp16 is not None:
                    speedup = f"{fp16 / best_us:5.2f}x"
                elif fp16 is not None and best_us is None:
                    best_name = "-"
                    best_us = fp16
                    speedup = "     -"
                else:
                    speedup = "   N/A"

                best_str = best_name if best_name else "N/A"

                print(
                    f"  | {shape:<6} | {k:>3} | {fmt(m_us)} | {fmt(s_us):>6} | {fmt(g_us):>7} | {fmt(gm_us):>7} | {fmt(fp16)} | {best_str:>7} | {speedup:>7} |"
                )

            print(f"  {HDR}")

        # Per-k total rows: sum best_us and fp16 across all shapes for each k
        print(f"  | {'TOTAL':<6} |     |       |        |         |         |       |        |         |")
        for k in k_bits:
            k_best = 0.0
            k_fp16 = 0.0
            k_complete = True
            for shape in all_shapes:
                is_moe = shape in moe_shapes
                m_us = mma.get((shape, k, M)) if not is_moe else None
                s_us = scalar.get((shape, k, M)) if not is_moe else None
                g_us = grouped.get((shape, k, M)) if is_moe else None
                gm_us = grouped_mma.get((shape, k, M)) if is_moe else None
                fp16 = cublas_moe.get((shape, M)) if is_moe else cublas_dense.get((shape, M))

                candidates = {}
                if m_us is not None:
                    candidates["MMA"] = m_us
                if s_us is not None:
                    candidates["Scalar"] = s_us
                if g_us is not None:
                    candidates["Grouped"] = g_us
                if gm_us is not None:
                    candidates["Grp MMA"] = gm_us

                if candidates:
                    best_us = min(candidates.values())
                elif fp16 is not None:
                    best_us = fp16
                else:
                    k_complete = False
                    continue

                k_best += best_us
                if fp16 is not None:
                    k_fp16 += fp16

            if k_complete and k_best > 0 and k_fp16 > 0:
                overall = k_fp16 / k_best
                print(
                    f"  |  k={k:<3} | {k:>3} |       |        |         |         |       | {k_best:6.1f} | {overall:5.2f}x  |"
                )
            else:
                print(f"  |  k={k:<3} | {k:>3} |       |        |         |         |       |   N/A  |    N/A  |")
        print(f"  {TOP}")


if __name__ == "__main__":
    main()
