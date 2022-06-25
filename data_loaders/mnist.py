import torchvision
from torch.utils.data import DataLoader

__all__ = ['MNIST']

def MNIST(split, root, batch_size, num_workers=1, **kwargs):
    dataset = torchvision.datasets.MNIST(
            root=root,
            train=(split=='train'),                         
            transform=torchvision.transforms.ToTensor(), 
            download=True)
    loader = DataLoader(dataset=dataset, num_workers=num_workers, 
                        batch_size=batch_size, **kwargs)
    return loader
    