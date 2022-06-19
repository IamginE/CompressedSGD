from tqdm import tqdm
from torch import nn
import torch
import numpy as np
from .optimizers.compressed_sgd import CompressedSGD
cuda_available = torch.cuda.is_available()
class Trainer():
    def __init__(self, model, train_loader, eval_loader, optimizer, batchwise_evaluation=False,
                 plot=True, num_workers=1, **kwargs):
        super(Trainer, self).__init__()
        self.model = model.cuda() if cuda_available else model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.optimizer = optimizer
        self.batchwise_evaluation = batchwise_evaluation
        self.plot = plot
        self.num_workers = num_workers
        self.loss = nn.CrossEntropyLoss()
        if hasattr(self.optimizer, 'is_cumulative') and self.optimizer.is_cumulative:
            assert hasattr(self.optimizer, 'aggregate'), 'Cumulative optimizer without accumulate!'
            self.batch_action = self.optimizer.aggregate
            self.epoch_action = self.optimizer.step
        else:
            self.batch_action = lambda *args: self.optimizer.step()
            self.epoch_action = lambda *args: None #Do nothing
        
    def train(self, epochs):
        self.model.train()
        avg_epoch_acc_hist = [] # Average of batch_losses for each epoch
        avg_epoch_loss_hist = [] # Average of batch_accs for each epoch
        batch_loss_hist = [] # Exact loss over the entire dataset after each batch
        batch_acc_hist = [] # Exact acc over the entire dataset after each batch
        for ep in range(epochs):
            num_correct = 0
            num_inputs = 0
            total_loss = 0.0
            for batch_idx, (inputs, targets) in \
                tqdm(enumerate(self.train_loader), desc='Epoch', total=len(self.train_loader)):
                batch_size = inputs.size(0)
                _inputs = inputs.cuda() if cuda_available else inputs
                _targets = targets.cuda() if cuda_available else targets
                out = self.model(_inputs)

                loss = self.loss(out, _targets)
                preds = out.argmax(dim=1)
                curr_correct = (preds == _targets).sum().item()
                num_correct += curr_correct
                curr_loss = loss.item()
                total_loss += curr_loss * batch_size
                num_inputs += batch_size

                self.optimizer.zero_grad()
                loss.backward()

                self.batch_action(batch_idx%self.num_workers)
                if batch_idx%self.num_workers == self.num_workers-1:
                    self.epoch_action()
                    if self.batchwise_evaluation:
                        current_loss, current_acc = self.evaluate()
                        batch_loss_hist.append(current_loss)
                        batch_acc_hist.append(current_acc)
                    else:
                        batch_loss_hist.append(curr_loss)
                        batch_acc_hist.append(curr_correct/ float(batch_size) * 100)
            avg_epoch_loss_hist.append(total_loss / float(num_inputs))
            avg_epoch_acc_hist.append(num_correct / float(num_inputs) * 100)
        
        if isinstance(self.optimizer, CompressedSGD):
            return {'avg_epoch_loss_hist': avg_epoch_loss_hist, 'avg_epoch_acc_hist': avg_epoch_acc_hist, 
                'batch_loss_hist': batch_loss_hist, 'batch_acc_hist': batch_acc_hist, 
                'bin_usage': np.array(self.optimizer.bin_counts)}

        return {'avg_epoch_loss_hist': avg_epoch_loss_hist, 'avg_epoch_acc_hist': avg_epoch_acc_hist, 
                'batch_loss_hist': batch_loss_hist, 'batch_acc_hist': batch_acc_hist}
    
    def evaluate(self):
        self.model.eval()
        num_correct = 0
        total_loss = 0.0
        num_inputs = 0
        for batch_idx, (inputs, targets) in \
            tqdm(enumerate(self.eval_loader), desc='Evaluate', total=len(self.eval_loader)):
            out = self.model(inputs)

            loss = self.loss(out, targets)
            preds = out.argmax(dim=1)
            num_correct += (preds == targets).sum().item()
            total_loss += loss.item() * inputs.size(0)
            num_inputs += inputs.size(0)
        total_loss /= float(num_inputs)
        accuracy = (num_correct / float(num_inputs)) * 100
        self.model.train()
        return total_loss, accuracy