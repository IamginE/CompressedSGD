import optimizers
import models
import data_loaders
import torch
import numpy as np
import pandas as pd
import random
import pickle
from trainer import Trainer
from config import get_configs
from os.path import join as ospj
import numpy as np
from matplotlib import pyplot as plt

def _plot(histories, args):
    print(histories)
    plt.subplots(squeeze = False)


    file = 'sample.png'
    metrics = histories[(args.lr[0], args.weight_decay[0])].keys()
    for metric in metrics:
        if metric == 'bin_usage' and args.count_usages:
            fig, axs = plt.subplots(len(args.lr), len(args.weight_decay), figsize=(20,15), squeeze=False)

            for i in range(len(args.lr)):
                for j in range(len(args.weight_decay)):
                    df = pd.DataFrame(histories[(args.lr[i], args.weight_decay[j])][metric], columns = [i for i in range (-2**(args.num_bits-1), 2**(args.num_bits-1)+1)])
                    df.plot(kind='bar', stacked=True, ax = axs[i, j], title = f'bin usage, lr: {args.lr[i]}, beta: {args.weight_decay[j]}, binning: {args.binning}')

            plt.tight_layout()
            plt.savefig(ospj(args.log_folder,metric+'.png'))

        else:
            sample_history = histories[(args.lr[0], args.weight_decay[0])][metric]
            x = np.linspace(1, len(sample_history),
                        num = len(sample_history))
            fig, axs = plt.subplots(len(args.lr), len(args.weight_decay), figsize=(20, 15), squeeze=False)
            for i in range(len(args.lr)):
                for j in range(len(args.weight_decay)):
                    axs[i, j].plot(x, histories[(args.lr[i], args.weight_decay[j])][metric])
                    # axs[i, j].set_ylim([0, 4])
                    axs[i, j].title.set_text(f'lr: {args.lr[i]}, beta: {args.weight_decay[j]}')

            plt.tight_layout()
            plt.savefig(ospj(args.log_folder,metric+'.png'))

def main():
    args = get_configs()
    print(args)

    # set random seeds
    if args.seed > 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    train_loader = data_loaders.__dict__[args.dataset](
            root=args.dataset_root,
            split='train',
            batch_size=args.batch_size,
            num_workers=4
            )
    eval_loader = data_loaders.__dict__[args.dataset](
            root=args.dataset_root,
            split='eval',
            batch_size=args.batch_size,
            num_workers=4
            )

    histories = {}
    for lr in args.lr:
        for weight_decay in args.weight_decay:
            print(f'Training with lr={lr}, weight_decay={weight_decay}')
            model = models.__dict__[args.model](
                num_classes=args.num_classes,
                pretrianed=args.pretrained
                )
            optimizer = optimizers.__dict__[args.optimizer](
                    params=model.parameters(),
                    lr=lr,
                    lr_decay=args.lr_decay,
                    decay_min=weight_decay,
                    decay_max=weight_decay,
                    num_bits=args.num_bits, 
                    rand_zero=args.rand_zero,
                    num_workers=args.num_workers,
                    binning=args.binning,
                    count_usages=args.count_usages
                )
            trainer = Trainer(model=model,
                    train_loader=train_loader,
                    eval_loader=train_loader,#eval_loader
                    optimizer=optimizer,
                    num_workers=args.num_workers,
                    batchwise_evaluation=args.batchwise_evaluation,
                )
            history = trainer.train(args.epochs)
            histories[(lr, weight_decay)] = history
    with open(ospj(args.log_folder, 'histories.pkl'), 'wb') as f:
        pickle.dump(histories, f)
    _plot(histories, args)
if __name__=='__main__':
    main()