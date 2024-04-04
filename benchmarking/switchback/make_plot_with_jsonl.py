import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd

cmap = plt.get_cmap("cool")

if __name__ == "__main__":
    fig = plt.figure(tight_layout=True, figsize=(12, 3.5))
    gs = gridspec.GridSpec(1, 2)

    dims_to_consider = [1024, 1280, 1408, 1664, 2048, 4096]
    batch_size_for_plot1 = 32768
    batch_sizes_for_plot2 = [2**14, 2**15, 2**16, 2**17]
    dims_to_xtick = [1024, 2048, 4096]
    logscale_plot1 = True

    ax = fig.add_subplot(gs[0, 0])

    # TODO: change this to what you want.
    rdf = pd.read_json("speed_benchmark/info_a100_py2.jsonl", lines=True)
    df = rdf[rdf.batch_size == batch_size_for_plot1]

    # first plot the time occupied by different operations
    for k, marker, ls, color, name in [
        ("standard_gx+standard_gw+standard_fwd", "s", "-", "C2", "Standard fp16 (sum of parts)"),
        (
            "x_quantize_rowwise+g_quantize_rowwise+w_quantize_global+w_quantize_global_transpose+standard_gw+global_fwd+global_bwd",
            "o",
            "-",
            "C4",
            "SwitchBack int8 (sum of parts)",
        ),
        ("standard_fwd", "^", "--", "C2", "Matmul XW (standard)"),
        ("standard_gw", "^", "-.", "C2", "Matmul GW (standard)"),
        ("standard_gx", "^", ":", "gray", "Matmul GX (both)"),
        ("global_fwd", "^", "--", "C4", "Int8 Matmul XW (switchback)"),
        ("global_bwd", "^", "-.", "C4", "Int8 Matmul GW (switchback)"),
        ("x_quantize_rowwise", "P", "--", "C4", "Quantize rowwise X (switchback)"),
        ("g_quantize_rowwise", "P", "-.", "C4", "Quantize rowwise G (switchback)"),
        ("w_quantize_global", ".", "--", "C4", "Quantize global W (switchback)"),
        ("w_quantize_global_transpose", ".", "-.", "C4", "Quantize global and\ntranspose W (switchback)"),
    ]:
        xs = []
        ys = []
        for embed_dim in dims_to_consider:
            # average over dim -> 4*dim and 4*dim -> dim
            df_ = df[df.dim_in == embed_dim]
            df_ = df_[df_.dim_out == embed_dim * 4]
            xs.append(embed_dim)
            y_ = 0
            for k_ in k.split("+"):
                y_ += df_[k_].values[0]
            df_ = df[df.dim_in == embed_dim * 4]
            df_ = df_[df_.dim_out == embed_dim]
            for k_ in k.split("+"):
                y_ += df_[k_].values[0]
            ys.append(y_ * 0.5)

        ax.plot(
            xs,
            ys,
            color=color,
            label=name,
            marker=marker,
            markersize=5 if marker == "s" else 5,
            linestyle=ls,
            linewidth=2 if "+" in k else 1.0,
        )

    ax.set_xlabel("dim", fontsize=13)
    ax.set_ylabel("time (ms)", fontsize=13)

    ax.grid()

    ax.set_xscale("log")
    if logscale_plot1:
        ax.set_yscale("log")

    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)

    ax.set_xticks(dims_to_xtick)
    ax.set_xticklabels(dims_to_xtick)
    ax.set_xticks([], minor=True)

    leg = ax.legend(loc="upper center", bbox_to_anchor=(-0.64, 1.0), ncol=1, fontsize=10)
    leg.get_texts()[0].set_fontweight("bold")
    leg.get_texts()[1].set_fontweight("bold")
    plt.subplots_adjust(left=0.1)
    ax.set_title("  Linear layer, batch * sequence length = 32k", fontsize=10, loc="left", y=1.05, pad=-20)

    ax = fig.add_subplot(gs[0, 1])

    # now plot the % speedup for different batch sizes
    for j, batch_size in enumerate(batch_sizes_for_plot2):
        all_xs, all_ys = [], []
        for k, marker, ls, color, name in [
            ("standard_gx+standard_gw+standard_fwd", "s", "-", "C2", "Standard fp16 (total time)"),
            (
                "x_quantize_rowwise+g_quantize_rowwise+w_quantize_global+w_quantize_global_transpose+standard_gw+global_fwd+global_bwd",
                "o",
                "-",
                "C4",
                "SwitchBack int8 (total time)",
            ),
        ]:
            xs, ys = [], []
            df = rdf[rdf.batch_size == batch_size]
            for embed_dim in dims_to_consider:
                df_ = df[df.dim_in == embed_dim]
                df_ = df_[df_.dim_out == embed_dim * 4]
                xs.append(embed_dim)
                y_ = 0
                for k_ in k.split("+"):
                    y_ += df_[k_].values[0]
                df_ = df[df.dim_in == embed_dim * 4]
                df_ = df_[df_.dim_out == embed_dim]
                for k_ in k.split("+"):
                    y_ += df_[k_].values[0]
                ys.append(y_ * 0.5)
            all_xs.append(xs)
            all_ys.append(ys)

        color = cmap(j * 0.25)
        real_ys = [-((all_ys[1][i] - all_ys[0][i]) / all_ys[0][i]) * 100 for i in range(len(all_ys[0]))]
        markers = ["^", "v", "P", "o"]
        ax.plot(
            all_xs[0],
            real_ys,
            color=color,
            label=f"batch * sequence length = {batch_size}",
            marker=markers[j],
            markersize=5 if marker == "s" else 5,
        )

    ax.legend()
    ax.set_xlabel("dim", fontsize=13)
    ax.set_xscale("log")
    ax.grid()
    ax.set_ylabel(r"% speedup", fontsize=13)

    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)

    ax.set_xticks(dims_to_xtick)
    ax.set_xticklabels(dims_to_xtick)
    ax.set_xticks([], minor=True)

    ax.set_title("  Linear layer summary, varying dimensions", fontsize=10, loc="left", y=1.05, pad=-20)

    plt.savefig("speed_benchmark/plot_with_info.pdf", bbox_inches="tight")
