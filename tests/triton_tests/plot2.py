import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

import matplotlib.gridspec as gridspec

cmap=plt.get_cmap('cool')

if __name__ == '__main__':

    fig = plt.figure(tight_layout=True, figsize=(6,3.5))
    gs = gridspec.GridSpec(1, 1)


    rdf = pd.read_json('tests/triton_tests/info.jsonl', lines=True)

    ax = fig.add_subplot(gs[0, 0])

    # now plot the % speedup for different batch sizes
    for j, batch_size in enumerate([2**14, 2**15, 2**16, 2**17]):
        all_xs, all_ys = [], []
        for k, marker, ls, color, name in [
            ('x_quantize_rowwise+g_quantize_rowwise+w_quantize_global+w_quantize_global_transpose+standard_gw+global_fwd+global_bwd', 'o', '-', 'C4', 'SwitchBack int8 (total time)'),
            ('x_quantize_rowwise+g_quantize_rowwise+w_quantize_global+w_quantize_global_transpose', 'o', '-', 'C4', 'SwitchBack int8 (total time)'),
        ]:
        
            xs, ys = [], []
            df = rdf[rdf.batch_size == batch_size]
            for embed_dim in [1024, 1280, 1408, 1664, 2048, 4096]:
                df_ = df[df.dim_in == embed_dim]
                df_ = df_[df_.dim_out == embed_dim * 4]
                xs.append(embed_dim)
                y_ = 0
                for k_ in k.split('+'):
                    y_ += df_[k_].values[0]
                df_ = df[df.dim_in == embed_dim * 4]
                df_ = df_[df_.dim_out == embed_dim]
                for k_ in k.split('+'):
                    y_ += df_[k_].values[0]
                ys.append(y_ * 0.5)
            all_xs.append(xs)
            all_ys.append(ys)

        color = cmap(j * 0.25)
        real_ys = [100 * all_ys[1][i] / all_ys[0][i] for i in range(len(all_ys[0]))]
        markers = ['^', 'v', 'P', 'o']
        ax.plot(all_xs[0], real_ys, color=color, label=f'batch * sequence length = {batch_size}', marker=markers[j], markersize=5 if marker=='s' else 5)

    ax.legend()
    ax.set_xlabel('dim', fontsize=13)
    ax.set_xscale('log')
    ax.grid()
    ax.set_ylabel(r'% time occupied by quantize ops', fontsize=12)


    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)

    ax.set_xticks([1024, 2048, 4096])
    ax.set_xticklabels([1024, 2048, 4096])
    ax.set_xticks([], minor=True)

    #ax.set_title('  Linear layer summary, varying dimensions', fontsize=10, loc='left', y=1.05, pad=-20)



    plt.savefig('tests/triton_tests/plot2.pdf', bbox_inches='tight')

