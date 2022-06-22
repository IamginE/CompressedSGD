import torch
import torch.optim as optim

__all__ = ['compressed_sgd_vote']

class CompressedSGDVote(optim.Optimizer):
  r"""Generalized gradient compression using binning.
  """
  def __init__(self, params, lr, rand_zero, num_bits, decay_max, 
               decay_min, num_workers, count_usages, binning='lin'):

    if num_bits <= 0 or type(num_bits) != int:
        raise ValueError("Expected num_bits to be a positive integer.")
    if not (0 < decay_max <= 1) or not ( 0 < decay_min <= 1):
        raise ValueError("Expected decay_max and decay_min to be a float in the interval (0,1].")
    if binning not in ["lin", "exp"]:
        raise ValueError("Binning method needs to be 'lin' (linear) or 'exp' (exponential).")

    defaults = dict(lr=lr)
    super(CompressedSGDVote, self).__init__(params, defaults)
    self.is_cumulative = True
    self.decay_max = decay_max
    self.decay_min = decay_min
    self.rand_zero = rand_zero
    self.max_grad_vals = [{} for i in range(num_workers)]
    self.min_grad_vals = [{} for i in range(num_workers)]
    self.grad_vals = [{} for i in range(num_workers)]
    self.used = [False for i in range(num_workers)]
    self.num_bits = num_bits
    self.num_workers = num_workers
    self.binning = binning
    self.count_usages = count_usages
    self.bin_counts = []
    
    for i in range(num_workers):
      for group in self.param_groups:
        for p in group['params']:
          self.max_grad_vals[i][p] = None
          self.min_grad_vals[i][p] = None
  
  def aggregate(self, worker_id):
    r"""Simulates distributed workers by aggregating gradients.
    Needs to be called after loss.backward()!"""
    if worker_id < 0 or worker_id >= self.num_workers:
      raise ValueError("Worker_id needs to be an integer in the interval [0, num_workers)")
    
    if self.used[worker_id]:
      print("Warning: Already aggregated batch gradients for this worker and no voting was initilized. Overwriting old gradient!")

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue

        # save the gradients for voting
        self.grad_vals[worker_id][p] = torch.clone(p.grad)

        # set the max and min vals for gradients
        if self.max_grad_vals[worker_id][p] == None:
          self.max_grad_vals[worker_id][p] = torch.clone(p.grad)
        else:
          self.max_grad_vals[worker_id][p] = torch.max(p.grad, 
                                            self.max_grad_vals[worker_id][p] * self.decay_max)

        if self.min_grad_vals[worker_id][p] == None:
          self.min_grad_vals[worker_id][p] = torch.clone(p.grad)
        else:
          self.min_grad_vals[worker_id][p] = torch.min(p.grad, 
                                            self.min_grad_vals[worker_id][p] * self.decay_min)
    
    self.used[worker_id] = True

  def step(self, closure=None):
    r"""Performs a single optimization step.
    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    flag = False
    for i in range(self.num_workers):
      flag = flag or self.used[i]
    assert flag

    loss = None
    if closure is not None:
      loss = closure()

    # parameter update
    bin_count = torch.tensor([0 for i in range(2**(self.num_bits) + 1)])
    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue

        # aggregate the saved gradients and simulate a voting process
        aggr_grad = None
        count = 0
        for i in range(self.num_workers):
            
            if not self.used[i]:
                continue
          
            count += 1
            # descretize gradient based on max_grad_val, if gradient > 0
            # and based on min_grad_val, if gradient < 0
            grad = self.grad_vals[i][p]
            if self.binning == 'lin':
                grad[grad > 0] = torch.ceil((2**(self.num_bits-1)) * torch.div(grad[grad > 0], self.max_grad_vals[i][p][grad > 0]))
                grad[grad < 0] = - torch.ceil((2**(self.num_bits-1)) * torch.div(grad[grad < 0], self.min_grad_vals[i][p][grad < 0]))
            elif self.binning == 'exp':
                grad[grad > 0] = torch.ceil((2**(self.num_bits-1)) * (torch.div(torch.log(grad[grad > 0] + 1.0001), torch.log(self.max_grad_vals[i][p][grad > 0] + 1.0001)))) # log_e(x+1)/log_e(maxgrad+1) = log_(maxgrad+1)(x+1)
                grad[grad < 0] = -torch.ceil((2**(self.num_bits-1)) * (torch.div(torch.log(-grad[grad < 0] + 1.0001), torch.log(-self.min_grad_vals[i][p][grad < 0] + 1.0001))))

            if self.rand_zero:
              grad[grad==0] = torch.randint_like(grad[grad==0], low=0, high=2)*2 - 1
            aggr_grad = torch.clone(grad) if aggr_grad is None else torch.add(aggr_grad, grad)

        # simulate the voting
        aggr_grad[aggr_grad > 0] = torch.ceil(torch.div(aggr_grad[aggr_grad > 0], torch.tensor(count)))
        aggr_grad[aggr_grad < 0] = torch.floor(torch.div(aggr_grad[aggr_grad < 0], torch.tensor(count)))
        if self.rand_zero:
          aggr_grad[aggr_grad==0] = torch.randint_like(aggr_grad[aggr_grad==0], low=0, high=2)*2 - 1
          
        # make update
        p.data -= group['lr'] * aggr_grad
        if self.count_usages:
          bin_count = torch.add(bin_count, torch.bincount(torch.flatten(torch.clone(aggr_grad).cpu().type(torch.int64)) + 2**(self.num_bits-1), minlength=2**self.num_bits+1))

    self.used = [False for i in range(self.num_workers)]

    if self.count_usages:
      self.bin_counts.append(bin_count.numpy())
    return loss

def compressed_sgd_vote(params, lr, rand_zero, num_bits,
                decay_max, decay_min, num_workers, binning, count_usages, **kwargs):
    return CompressedSGDVote(params=params, lr=lr, rand_zero=rand_zero,
                num_bits=num_bits, decay_max=decay_max, decay_min=decay_min, num_workers=num_workers, count_usages=count_usages, binning=binning)