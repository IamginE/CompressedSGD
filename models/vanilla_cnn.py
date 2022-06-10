import torch.nn.functional as F
import torch
from torch import nn


model_url = None
__all__ = ['vanilla_cnn']

class VanillaCnn(nn.Module):
  def __init__(self, num_classes):
    super(VanillaCnn, self).__init__()
    # Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    # padding, so the produced convolutions have orginal size
    self.conv = nn.Conv2d(1, 32, 3, 1, 1, padding_mode = 'zeros')

    # MaxPool2D(kernel_size, stride)
    self.maxpool = nn.MaxPool2d(2, stride=2) # stride's default value is kernel size

    self.dropout1 = nn.Dropout(p=0.5)
    self.fc1 = nn.Linear(14*14*32, 512)
    self.dropout2 = nn.Dropout(p=0.5)
    self.fc2 = nn.Linear(512, num_classes)
  
  def forward(self, x):
      x = self.conv(x)
      x = F.relu(x)
      x = self.maxpool(x)
      x = torch.flatten(x,1)
      x = self.dropout1(x)
      x = self.fc1(x)
      x = F.relu(x)
      x = self.dropout2(x)
      x = self.fc2(x)
      return(x)

def load_pretrained_model(model, url):
    raise NotImplementedError

def vanilla_cnn(num_classes, pretrained=False, **kwargs):
    model = VanillaCnn(num_classes)
    if pretrained:
        load_pretrained_model(model, model_url)
    return model
