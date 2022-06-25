import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""if __name__=='__main__':
    plot('batch_loss_hist', 
        files=['./logs/sign_sgd_B64_e1_EvalInterval50_Seed1917_2/histories.pkl',
        './logs/sign_sgd_B64_e1_EvalInterval50_Seed1917/histories.pkl',
        './logs/compressed_sgd_B64_n3_llin1_EvalInterval50_Seed1917_4/histories.pkl',
        './logs/compressed_sgd_B64_n3_llin1_EvalInterval50_Seed1917_4/histories.pkl',
        ],
        keys=[(0.0015, 1), (0.001, 1),
            (0.0007, 0.9), (0.0008, 0.9)
            ],
        names=['signSGD lr=0.0015', 'signSGD lr=0.001',
                'compressedSGD 3bits lr=0.0007, decay=0.9', 'compressedSGD 3bits lr=0.0008, decay=0.9'
                ],
        out='cSGD_3bits_better_sign.png',
        linspace_mul=50)"""

def plot(metric, files, keys, names, out, title, xaxis, yaxis, linspace_mul=1):
    r""""Plots multiple line plots for the specified files and metric in one plot int the output file 'out'.
    Uses the produced dictionaries from trainer.py."""
    fig, axs = plt.subplots(1, 1, figsize=(10, 8), squeeze=False)
    for k in range(len(files)):
        with open(files[k],'rb') as f:
            histories = pickle.load(f)

        sample_history = histories[keys[k]][metric]
        x = linspace_mul*np.linspace(1, len(sample_history), num = len(sample_history))

        label = names[k]
        axs[0,0].plot(x, histories[keys[k]][metric], label=label)
        axs[0,0].legend(loc="best")
        axs[0,0].set_title(title)
        axs[0,0].set_xlabel(xaxis)
        axs[0,0].set_ylabel(yaxis)


    plt.tight_layout()
    plt.savefig(out)



def plot_usages(files, num_x_plots, num_y_plots, figsize, num_bits, keys, out, titles):
    r""""Plots multiple bin usages in one picture, to output file 'out'.
    Uses the produced dictionaries from trainer.py."""
    fig, axs = plt.subplots(num_y_plots, num_x_plots, figsize=figsize, squeeze=False)
    for y in range(num_y_plots):
        for x in range(num_x_plots):
            k = x + y*num_x_plots
            with open(files[k],'rb') as f:
                histories = pickle.load(f)

            df = pd.DataFrame(histories[keys[k]]['bin_usage'], columns = [i for i in range (-2**(num_bits-1), 2**(num_bits-1)+1)])
            df.plot(kind='bar', stacked=True, ax = axs[y, x], xticks = [k*100 for k in range(np.shape(histories[keys[k]]['bin_usage'])[0]//100+1)], title =titles[k])
            axs[y,x].set_xlabel("update steps")
            axs[y,x].set_ylabel("bin usages summed over all parameters")
            axs[y,x].legend(loc='right')

    plt.tight_layout()
    plt.savefig(out)

def plot_avg(metric, files, keys, names, out, title, xaxis, yaxis, linspace_mul=1):
    r""""Plots multiple line plots for the specified files and metric in one plot int the output file 'out'.
    Here keys have to be the same for all files, because results are avereged over the different files for the same key.
    Uses the produced dictionaries from trainer.py."""
    fig, axs = plt.subplots(1, 1, figsize=(10, 8), squeeze=False)
    
    for i in range(len(keys)):
        sample_history = 0
        for k in range(len(files)):
            with open(files[k],'rb') as f:
                histories = pickle.load(f)

            sample_history += np.asarray(histories[keys[i]][metric])

        sample_history /= 3
        x = linspace_mul*np.linspace(1, len(sample_history), num = len(sample_history))
        label = names[i]
        axs[0,0].plot(x, sample_history, label=label)
        axs[0,0].legend(loc="best")
        axs[0,0].set_title(title)
        axs[0,0].set_xlabel(xaxis)
        axs[0,0].set_ylabel(yaxis)


        plt.tight_layout()
        plt.savefig(out)

