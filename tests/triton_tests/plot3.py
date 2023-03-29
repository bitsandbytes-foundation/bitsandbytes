import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec

cmap=plt.get_cmap('cool')

if __name__ == '__main__':

    fig = plt.figure(tight_layout=True, figsize=(12,3.5))
    gs = gridspec.GridSpec(1, 3)


    rdf1 = pd.read_json('tests/triton_tests/info_mlp_autocast_ln.jsonl', lines=True)

    ax = fig.add_subplot(gs[0, 0])

    # now plot the % speedup for different batch sizes
    for j, batch_size in enumerate([2**15, 2**17]):#, 2**15, 2**17, 2**17]):
        all_xs, all_ys = {}, {}
        for k, marker, ls, color, name in [
            ('standard_compiled', 'o', '-', 'C0', 'standard compiled (total time)'),
            #('standard', 'o', '-', 'C1', 'standard (total time)'),
            ('my_standard', 'o', '-', 'C2', 'my standard (total time)'),
            ('sb', 'o', '-', 'C4', 'SwitchBack int8 (total time)'),
        ]:
        
            xs, ys = [], []
            df = rdf1[rdf1.batch_size == batch_size]
            for embed_dim in [1024, 1280, 1408, 1664, 2048]:
                df_ = df[df.dim == embed_dim]
                xs.append(embed_dim)
                y_ = 0
                for k_ in k.split('+'):
                    y_ += df_[k_].values[0]
                ys.append(y_)

            all_xs[k] = xs
            all_ys[k] = ys
            #ax.plot(xs, ys, color=color, label=f'batch * sequence length = {batch_size}', marker=marker, markersize=5 if marker=='s' else 5)
        

        color= cmap(float(j))
        speedup_over_my_standard = [-100 * (all_ys['sb'][i] - all_ys['my_standard'][i]) / all_ys['my_standard'][i] for i in range(len(all_ys['my_standard']))]
        speedup_over_compile = [-100 * (all_ys['sb'][i] - all_ys['standard_compiled'][i]) / all_ys['standard_compiled'][i] for i in range(len(all_ys['standard_compiled']))]

        ax.plot(xs, speedup_over_my_standard, color=color, label=f'batch * sequence length = {batch_size}', marker='o', markersize=5 if marker=='s' else 5)
        ax.plot(xs, speedup_over_compile, color=color, label=f'batch * sequence length = {batch_size}', marker='o', markersize=5 if marker=='s' else 5, linestyle='--')


    #ax.legend()
    ax.set_xlabel('dim', fontsize=13)
    ax.set_xscale('log')
    ax.grid()
    ax.set_ylabel(r'% speedup', fontsize=12)

    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)

    ax.set_xticks([1024, 2048])
    ax.set_xticklabels([1024, 2048])
    ax.set_xticks([], minor=True)
    ax.set_title('MLP Block', fontsize=10, loc='left', y=1.07, pad=-20)


    ##########################################

    rdf2 = pd.read_json('tests/triton_tests/attn_info_ln.jsonl', lines=True)

    ax = fig.add_subplot(gs[0, 1])

    for j, batch_size in enumerate([2**15, 2**17]):#, 2**15, 2**17, 2**17]):
        all_xs, all_ys = {}, {}
        for k, marker, ls, color, name in [
            ('standard_compiled', 'o', '-', 'C0', 'standard compiled (total time)'),
            #('standard', 'o', '-', 'C1', 'standard (total time)'),
            ('my_standard', 'o', '-', 'C2', 'my standard (total time)'),
            ('sb', 'o', '-', 'C4', 'SwitchBack int8 (total time)'),
        ]:
        
            xs, ys = [], []
            df = rdf2[rdf2.batch_size == batch_size]
            for embed_dim in [1024, 1280, 1408, 1664, 2048]:
                df_ = df[df.dim == embed_dim]
                xs.append(embed_dim)
                y_ = 0
                for k_ in k.split('+'):
                    y_ += df_[k_].values[0]
                ys.append(y_)

            all_xs[k] = xs
            all_ys[k] = ys
            #ax.plot(xs, ys, color=color, label=f'batch * sequence length = {batch_size}', marker=marker, markersize=5 if marker=='s' else 5)
        
        color= cmap(float(j))
        speedup_over_my_standard = [-100 * (all_ys['sb'][i] - all_ys['my_standard'][i]) / all_ys['my_standard'][i] for i in range(len(all_ys['my_standard']))]
        speedup_over_compile = [-100 * (all_ys['sb'][i] - all_ys['standard_compiled'][i]) / all_ys['standard_compiled'][i] for i in range(len(all_ys['standard_compiled']))]

        ax.plot(xs, speedup_over_my_standard, color=color, label=f'batch * sequence length = {batch_size}', marker='o', markersize=5 if marker=='s' else 5)
        ax.plot(xs, speedup_over_compile, color=color, label=f'batch * sequence length = {batch_size}', marker='o', markersize=5 if marker=='s' else 5, linestyle='--')


    speedup_compiled = mlines.Line2D([], [], linestyle='--', color='gray', label='speedup over compiled')
    speedup_baseline = mlines.Line2D([], [], linestyle='-', color='gray', label='speedup over baseline')
    batch_size_4 = mlines.Line2D([], [], linestyle='-', color=cmap(0.), label=f'batch = {int(2**15 // 256)}, sequence = {256}')
    batch_size_8 = mlines.Line2D([], [], linestyle='-', color=cmap(1.), label=f'batch = {int(2**17 / 256)} sequence = {256}')

    # Create the legend with the proxy artists
    
    # adjust plots so that they dont get squished by putting the legend under both

    
    plt.subplots_adjust(left=0.2)
    plt.subplots_adjust(right=0.8)

    fig.legend(handles=[speedup_compiled, speedup_baseline, batch_size_4, batch_size_8], ncol=2, loc='upper center', bbox_to_anchor=(0.35, 0.255))

    ax.set_xlabel('dim', fontsize=13)
    ax.set_xscale('log')
    ax.grid()
    ax.set_ylabel(r'% speedup', fontsize=12)

    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)

    ax.set_xticks([1024, 2048])
    ax.set_xticklabels([1024, 2048])
    ax.set_xticks([], minor=True)

    ax.set_title('Attention Block', fontsize=10, loc='left', y=1.07, pad=-20)



    ##########################################

    

    ax = fig.add_subplot(gs[0, 2])

    for j, batch_size in enumerate([2**15]):#, 2**15, 2**17, 2**17]):
        all_xs, all_ys = {}, {}
        for k, marker, ls, color, name, b in [
            ('standard_compiled', 'o', '-', 'C0', 'standard compiled (total time)', False),
            ('standard_compiled', 'o', '-', 'C0', 'standard compiled (total time)', True),

            #('standard', 'o', '-', 'C1', 'standard (total time)'),
            #('my_standard', 'o', '-', 'C2', 'my standard (total time)'),
            ('attn', 'o', '-', 'C4', 'SwitchBack int8 (total time)', True),
        ]:
            rdf = rdf2 if b else rdf1
        
            xs, ys = [], []
            df = rdf[rdf.batch_size == batch_size]
            for embed_dim in [1024, 1280, 1408, 1664, 2048]:
                df_ = df[df.dim == embed_dim]
                xs.append(embed_dim)
                y_ = 0
                for k_ in k.split('+'):
                    y_ += df_[k_].values[0]
                ys.append(y_)

            all_xs[k + str(int(b))] = xs
            all_ys[k + str(int(b))] = ys
            #ax.plot(xs, ys, color=color, label=f'batch * sequence length = {batch_size}', marker=marker, markersize=5 if marker=='s' else 5)
        

        print(all_ys.keys())
        all_ys['standard_compiled'] = [x + y for x, y in zip(all_ys['standard_compiled0'], all_ys['standard_compiled1'])]

        speedup_over_my_standard = [100 * all_ys['attn1'][i] / (all_ys['standard_compiled'][i] + all_ys['attn1'][i]) for i in range(len(all_ys['standard_compiled']))]
        ax.plot(xs, speedup_over_my_standard, color='gold', label=r'% time occupied by attention', marker='H', markersize=8)

        speedup_over_my_standard = [100 * all_ys['standard_compiled1'][i] / (all_ys['standard_compiled0'][i] + all_ys['standard_compiled1'][i]) for i in range(len(all_ys['standard_compiled']))]
        ax.plot(xs, speedup_over_my_standard, color='indianred', label=r'% time occupied by attention block', marker='P', markersize=8)


    ax.legend(bbox_to_anchor=(1.02, -0.27))
    ax.set_xlabel('dim', fontsize=13)
    ax.set_xscale('log')
    ax.grid()
    ax.set_ylabel(r'% time', fontsize=12)

    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)

    ax.set_xticks([1024, 2048])
    ax.set_xticklabels([1024, 2048])
    ax.set_xticks([], minor=True)

    plt.savefig('tests/triton_tests/plot3.pdf', bbox_inches='tight')

