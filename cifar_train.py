from torchvision import transforms
from torchvision import datasets
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from dataloaders import get_cifar10_dataloader, get_cifar100_dataloader
import os
import sys
import numpy as np
from utils.gpu_utils import restrict_GPU_pytorch
restrict_GPU_pytorch(str(sys.argv[1]))
def get_pct_cifar10_dataloader(pct, batch_size=32):
    data_root='/tmp/public_dataset/pytorch'
    train = True
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    dataset = datasets.CIFAR10(
                    root=data_root, train=train, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]))
    n_classes = 10
    targets = np.array(dataset.targets)
    class_indices = [np.where(targets == i)[0] for i in range(n_classes)]
    train_idxs = [np.random.choice(x, size=int(pct*len(x)), replace=False) for x in class_indices]
    subset_idxs = np.concatenate(train_idxs)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, 
                            pin_memory=False, sampler=SubsetRandomSampler(subset_idxs)) 
    return dataloader

def get_pct_cifar100_dataloader(pct, batch_size=32):
    data_root='/tmp/public_dataset/pytorch'
    train = True
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    dataset = datasets.CIFAR10(
                    root=data_root, train=train, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]))
    n_classes = 100
    targets = np.array(dataset.targets)
    class_indices = [np.where(targets == i)[0] for i in range(n_classes)]
    train_idxs = [np.random.choice(x, size=int(pct*len(x)), replace=False) for x in class_indices]
    subset_idxs = np.concatenate(train_idxs)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, 
                            pin_memory=False, sampler=SubsetRandomSampler(subset_idxs)) 
    return dataloader

from train_model import train_f
from pytorch_playground.cifar.model import cifar10
from tqdm.notebook import tqdm
import torch
# Same as pcts used for mnist
pct_choices = [.7, .8, .9, 1]
pct_choices = [.4, .5, .6]
bs = 256 
dataset = 'cifar100'
for pct in tqdm(pct_choices):
    model = cifar10(128)
    train_dataloader = get_pct_cifar10_dataloader(pct, batch_size=bs)
    test_dataloader = get_cifar10_dataloader(train=False, batch_size=bs)
    losses = train_f(model, train_dataloader)
    torch.save(model.state_dict(), "./saved_models/cifar10/cifar10_cnn_" + str(pct))
    np.savetxt("./saved_models/cifar10/cifar10_cnn_" + str(pct) + "_training_losses", losses)

    
