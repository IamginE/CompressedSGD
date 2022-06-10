import torch
import torch.nn as nn
import torch.optim as optim

__all__ = ['sign_sgd']

class SignSGD(optim.Optimizer):
  r"""Original implementation of signSGD from 
      https://github.com/jxbz/signSGD/blob/master/signSGD_zeros.ipynb
  """

  def __init__(self, params, lr=0.01, rand_zero=True):
    defaults = dict(lr=lr)
    super(SignSGD, self).__init__(params, defaults)
    self.rand_zero = rand_zero

  def step(self, closure=None):
    r"""Performs a single optimization step.
    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    loss = None
    if closure is not None:
      loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue

        # take sign of gradient
        grad = torch.sign(p.grad)

        # randomise zero gradients to ±1
        if self.rand_zero:
          grad[grad==0] = torch.randint_like(grad[grad==0], low=0, high=2)*2 - 1
          assert not (grad==0).any()
        
        # make update
        p.data -= group['lr'] * grad

    return loss

def sign_sgd(params, lr, rand_zero=True, **kwargs):
    return SignSGD(params=params, lr=lr, rand_zero=rand_zero)