import argparse
from os.path import join as ospj
import os
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_configs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=-1, help="Custom seed for initialization. Set to -1 to disable it.", type=int)
    parser.add_argument('--rand_zero', default=True, type=str2bool)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--model', help='Currently just vanilla_cnn is accepted', type=str)
    parser.add_argument('--dataset', help='Currently just MNIST is accepted', type=str)
    parser.add_argument('--dataset_root', help='Root dir of the dataset. If not exist, it will be downloaded.',type=str)
    parser.add_argument('--optimizer', help='Choose from [sgd, sign_sgd, compressed_sgd, compressed_sgd_vote]', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', help='space-separated list of learning rates.', nargs='+',
                        type=float, default=[0.01, 0.001, 0.0005, 0.0001, 0.00005])
    parser.add_argument('--weight_decay', help='space-separated list of min-max decays introduced in the report', nargs='+',
                        type=float, default=[1.0] )
    #parser.add_argument('--lr_decay', type=float)
    parser.add_argument('--binning', help='Binning strategy for compressed_sgd. Either use "lin" or "exp".', type=str, default=None)
    parser.add_argument('--num_bits', help='Number of bits to use for binning.', default=2, type=int)
    parser.add_argument('--num_workers', help='Number of workers for voting. Only effects results when compressed_sgd_vote is used as optimzier', default=1, type=int)
    parser.add_argument('--batchwise_evaluation', help='After every worker has received this many batches, an evaluation on the entire training set will be done. Set it to -1 to disable evaluation', default=-1, type=int)
    parser.add_argument('--count_usages', help='Counts the bin usages and plots them at the end', default=False, type=str2bool)
    parser.add_argument('--one_plot_optimizer', default=False, type=str2bool)

    args = parser.parse_args()
    args.log_folder = configure_log_folder(args)
    args.num_classes = configure_num_classes(args)
    args.pretrained=False #Corrently not being used. 
    return args

def configure_log_folder(args):
    log_folder = ospj('logs', args.exp_name)

    if os.path.isdir(log_folder):
        raise RuntimeError("Experiment with the same name exists: {}"
                            .format(log_folder))
    os.makedirs(log_folder)
    return log_folder

def configure_num_classes(args):
    return {"MNIST": 10,
            }[args.dataset]
