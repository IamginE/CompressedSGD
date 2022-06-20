import pickle
import numpy as np
import matplotlib.pyplot as plt

def plot(metric, files, keys, names, out):
    fig, axs = plt.subplots(1, 1, figsize=(10, 8), squeeze=False)
    for k in range(len(files)):
        with open(files[k],'rb') as f:
            histories = pickle.load(f)

        sample_history = histories[keys[k]][metric]
        x = np.linspace(1, len(sample_history), num = len(sample_history))

        label = names[k] + ' (lr, decay)=' + str(keys[k])
        axs[0, 0].plot(x, histories[keys[k]][metric], label=label)
        axs[0,0].legend(loc="best")


    plt.tight_layout()
    plt.savefig(out)