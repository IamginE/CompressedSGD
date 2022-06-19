import torch.optim as optim

__all__ = ['sgd']

def sgd(params, lr, **kwargs):
    return optim.SGD(params=params, lr=lr)