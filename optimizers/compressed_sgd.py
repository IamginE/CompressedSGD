import torch
import torch.nn as nn
import torch.optim as optim

__all__ = ['compressed_sgd']

class compressedSGD(optim.Optimizer):
  r"""Generalized gradient compression using binning.
  """
  def __init__(self, params, lr=0.01, rand_zero=True, num_bits=2, decay_max=1.0, decay_min=1.0):

    if num_bits <= 0 or type(num_bits) != int:
        raise ValueError("Expected num_bits to be a positive integer.")
    if not (0 < decay_max <= 1) or not ( 0 < decay_min <= 1):
        raise ValueError("Expected decay_max and decay_min to be a float\
                            in the interval (0,1].")

    defaults = dict(lr=lr)
    super(compressedSGD, self).__init__(params, defaults)
    self.decay_max = decay_max
    self.decay_min = decay_min
    self.max_grad_vals = {}
    self.min_grad_vals = {}
    self.num_bits=num_bits

    for group in self.param_groups:
      for p in group['params']:
        self.max_grad_vals[p] = None
        self.min_grad_vals[p] = None

  def step(self, closure=None):
    r"""Performs a single optimization step.
    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    loss = None
    if closure is not None:
      loss = closure()

    # parameter update
    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue

        # set the max and min vals for gradients
        if self.max_grad_vals[p] is None:
            self.max_grad_vals[p] = torch.clone(p.grad)
        else:
            self.max_grad_vals[p] = torch.max(p.grad, 
                                            self.max_grad_vals[p] * self.decay_max)

        if self.min_grad_vals[p] is None:
            self.min_grad_vals[p] = torch.clone(p.grad)
        else:
            self.min_grad_vals[p] = torch.min(p.grad, 
                                            self.min_grad_vals[p] * self.decay_min)
        # descretize gradient based on max_grad_val, if gradient > 0
        # and based on min_grad_val, if gradient < 0

        grad = p.grad
        grad[grad > 0] = torch.ceil((2**(self.num_bits-1)) * torch.div(grad[grad > 0], self.max_grad_vals[p][grad > 0]))
        grad[grad < 0] = - torch.ceil((2**(self.num_bits-1)) * torch.div(grad[grad < 0], self.min_grad_vals[p][grad < 0]))
        # make update
        p.data -= group['lr'] * grad

    return loss

def compressed_sgd(params, lr=0.01, rand_zero=True, num_bits=2,
                decay_max=1.0, decay_min=1.0, **kwargs):
    return compressedSGD(params=params, lr=lr, rand_zero=rand_zero,
                num_bits=num_bits, decay_max=decay_max, decay_min=decay_min)