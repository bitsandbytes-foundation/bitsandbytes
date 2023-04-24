import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

import matplotlib.gridspec as gridspec

cmap=plt.get_cmap('cool')

if __name__ == '__main__':

    fig = plt.figure(tight_layout=True, figsize=(12,3.5))
    gs = gridspec.GridSpec(1, 2)


    ax = fig.add_subplot(gs[0, 0])

    rdf = pd.read_json('tests/triton_tests/info.jsonl', lines=True)
    df = rdf[rdf.batch_size == 32768]

    for k, marker, ls, color, name in [
        ('standard_gx+standard_gw+standard_fwd', 's', '-', 'C2', 'Standard fp16 (sum of parts)'),
        ('x_quantize_rowwise+g_quantize_rowwise+w_quantize_global+w_quantize_global_transpose+standard_gw+global_fwd+global_bwd', 'o', '-', 'C4', 'SwitchBack int8 (sum of parts)'),

        ('standard_fwd+standard_gw+standard_gx', '^', '--', 'C2', 'Average fp16 matmul'),

        ('global_fwd+global_bwd', 'v', '--', 'C4', 'Average int8 matmul'),
        
        ####                 time_global = info['x_quantize_rowwise'] + info['g_quantize_rowwise'] + info['w_quantize_global'] + info['w_quantize_global_transpose'] + info['standard_gw'] + info['global_fwd'] + info['global_bwd']

        ('x_quantize_rowwise+g_quantize_rowwise+w_quantize_global+w_quantize_global_transpose', 'x', ':', 'C4', 'Average quantize operation'),
        #('standard_gw', '.', '--', 'C1', 'standard_gw'),
    ]:
        xs = []
        ys = []
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
            
            if 'average' in name.lower():
                y_ = y_ / len(k.split('+'))
            ys.append(y_ * 0.5)

        
        ax.plot(xs, ys, color=color, label=name, marker=marker, markersize=6 if marker=='s' else 6, linestyle=ls, linewidth=2 if '+' in k else 1.)




    ax.set_xlabel('dim', fontsize=13)
    ax.set_ylabel('time (ms)', fontsize=13)
    # make a legend which is below the plot



    ax.grid()

    ax.set_xscale('log')
    #ax.set_yscale('log')
    
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)

    ax.set_xticks([1024, 2048, 4096])
    ax.set_xticklabels([1024, 2048, 4096])
    ax.set_xticks([], minor=True)

    leg = ax.legend(loc='upper center', bbox_to_anchor=(-0.64,  1.), ncol=1, fontsize=10)
    leg.get_texts()[0].set_fontweight('bold')
    leg.get_texts()[1].set_fontweight('bold')
    plt.subplots_adjust(left=0.1)
    ax.set_title('  Linear layer, batch * sequence length = 32k', fontsize=10, loc='left', y=1.05, pad=-20)


    ax = fig.add_subplot(gs[0, 1])

    # now plot the % speedup for different batch sizes
    for j, batch_size in enumerate([2**14, 2**15, 2**16, 2**17]):
        all_xs, all_ys = [], []
        for k, marker, ls, color, name in [
            ('standard_gx+standard_gw+standard_fwd', 's', '-', 'C2', 'Standard fp16 (total time)'),
            ('x_quantize_rowwise+g_quantize_rowwise+w_quantize_global+w_quantize_global_transpose+standard_gw+global_fwd+global_bwd', 'o', '-', 'C4', 'SwitchBack int8 (total time)'),
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
        real_ys = [-((all_ys[1][i] - all_ys[0][i]) / all_ys[0][i]) * 100 for i in range(len(all_ys[0]))]
        markers = ['^', 'v', 'P', 'o']
        ax.plot(all_xs[0], real_ys, color=color, label=f'batch * sequence length = {batch_size}', marker=markers[j], markersize=5 if marker=='s' else 5)

    ax.legend()
    ax.set_xlabel('dim', fontsize=13)
    ax.set_xscale('log')
    ax.grid()
    ax.set_ylabel(r'% speedup', fontsize=13)


    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)

    ax.set_xticks([1024, 2048, 4096])
    ax.set_xticklabels([1024, 2048, 4096])
    ax.set_xticks([], minor=True)

    ax.set_title('  Linear layer summary, varying dimensions', fontsize=10, loc='left', y=1.05, pad=-20)



    plt.savefig('tests/triton_tests/plot1.pdf', bbox_inches='tight')

