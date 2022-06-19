import torch
import torch.optim as optim

__all__ = ['compressed_sgd']

class CompressedSGD(optim.Optimizer):
  r"""Generalized gradient compression using binning.
  """

  def __init__(self, params, lr, rand_zero, num_bits, decay_max, 
               decay_min, count_usages, binning='lin'):

    if num_bits <= 0 or type(num_bits) != int:
       raise ValueError("Expected num_bits to be a positive integer.")
    if decay_max > 1 or decay_max <= 0 or decay_min > 1 or decay_min <= 0:
      raise ValueError("Expected decay_max and decay_min to be a float in the interval (0,1].")
    if binning != "lin" and binning != "exp":
      raise ValueError("Binning method needs to be 'lin' (linear) or 'exp' (exponential).")

    defaults = dict(lr=lr)
    super(CompressedSGD, self).__init__(params, defaults)
    self.decay_max = decay_max
    self.decay_min = decay_min
    self.rand_zero = rand_zero
    self.max_grad_vals = {}
    self.min_grad_vals = {}
    self.num_bits = num_bits
    self.binning = binning
    self.count_usages = count_usages
    self.bin_counts = []

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
    bin_count = torch.tensor([0 for i in range(2**(self.num_bits) + 1)])
    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue

        # set the max and min vals for gradients
        if self.max_grad_vals[p] == None:
          self.max_grad_vals[p] = torch.clone(p.grad)
        else:
          self.max_grad_vals[p] = torch.max(p.grad, 
                                            self.max_grad_vals[p] * self.decay_max)

        if self.min_grad_vals[p] == None:
          self.min_grad_vals[p] = torch.clone(p.grad)
        else:
          self.min_grad_vals[p] = torch.min(p.grad, 
                                            self.min_grad_vals[p] * self.decay_min)
        # descretize gradient based on max_grad_val, if gradient > 0
        # and based on min_grad_val, if gradient < 0

        grad = p.grad
        if self.binning == 'lin':
          grad[grad > 0] = torch.ceil((2**(self.num_bits-1)) * torch.div(grad[grad > 0], self.max_grad_vals[p][grad > 0]))
          grad[grad < 0] = - torch.ceil((2**(self.num_bits-1)) * torch.div(grad[grad < 0], self.min_grad_vals[p][grad < 0]))
        elif self.binning == 'exp':
          grad[grad > 0] = torch.ceil((2**(self.num_bits-1)) * (torch.div(torch.log(grad[grad > 0] + 1), torch.log(self.max_grad_vals[p][grad > 0] + 1)))) # log_e(x+1)/log_e(maxgrad+1) = log_(maxgrad+1)(x+1)
          grad[grad < 0] = -torch.ceil((2**(self.num_bits-1)) * (torch.div(torch.log(-grad[grad < 0] + 1), torch.log(-self.min_grad_vals[p][grad < 0] + 1))))
        
        # randomise zero gradients to ±1
        if self.rand_zero:
          grad[grad==0] = torch.randint_like(grad[grad==0], low=0, high=2)*2 - 1
          assert not (grad==0).any()
        # make update
        p.data -= group['lr'] * grad

        # update bin count
        if self.count_usages:
          bin_count = torch.add(bin_count, torch.bincount(torch.flatten(torch.clone(grad).cpu().type(torch.int64)) + 2**(self.num_bits-1), minlength=2**self.num_bits+1))

    if self.count_usages:
      self.bin_counts.append(bin_count.numpy())
    return loss


def compressed_sgd(params, lr, rand_zero, num_bits, decay_max, 
               decay_min, count_usages, binning='lin', **kwargs):
    return CompressedSGD(params=params, lr=lr, rand_zero=rand_zero,
                num_bits=num_bits, decay_max=decay_max, decay_min=decay_min, count_usages=count_usages, binning=binning)