import pickle
import numpy as np
import matplotlib.pyplot as plt

def plot(metric, files, keys, names, out, linspace_mul=1):
    fig, axs = plt.subplots(1, 1, figsize=(10, 8), squeeze=False)
    for k in range(len(files)):
        with open(files[k],'rb') as f:
            histories = pickle.load(f)

        sample_history = histories[keys[k]][metric]
        x = linspace_mul*np.linspace(1, len(sample_history), num = len(sample_history))

        label = names[k]
        axs[0, 0].plot(x, histories[keys[k]][metric], label=label)
        axs[0,0].legend(loc="best")


    plt.tight_layout()
    plt.savefig(out)

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
     linspace_mul=50)