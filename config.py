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

    # parser.add_argument('--seed', type=int)
    parser.add_argument('--pretrained', default=False, type=str2bool)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--step_size', type=float)
    parser.add_argument('--lr_decay', type=float)
    parser.add_argument('--decay_min', default=1.0, type=float)
    parser.add_argument('--decay_max', default=1.0, type=float)
    parser.add_argument('--num_bits', default=2, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    args = parser.parse_args()
    args.log_folder = configure_log_folder(args)
    args.num_classes = configure_num_classes(args)
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
