from tqdm import tqdm
from torch import nn

class Trainer():
    def __init__(self, model, data_loader, optimizer, **kwargs):
        super(Trainer, self).__init__()
        self.model = model
        self.loader = data_loader
        self.optimizer = optimizer
        self.loss = nn.CrossEntropyLoss()

    def train(self, epochs):
        acc_hist = []
        loss_hist = []
        for ep in range(epochs):
            num_correct = 0
            num_inputs = 0
            total_loss = 0.0

            for batch_idx, (inputs, targets) in \
                tqdm(enumerate(self.loader), desc='Epoch', total=len(self.loader)):
                out = self.model(inputs)

                loss = self.loss(out, targets)
                total_loss += loss.item() * inputs.size(0)

                preds = out.argmax(dim=1)
                num_correct += (preds == targets).sum().item()
                num_inputs += inputs.size(0)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            loss_hist.append(total_loss / float(num_inputs))
            acc_hist.append(num_correct / float(num_inputs) * 100)
        
        return {'loss': loss_hist, 'acc': acc_hist}