import os
import torch
import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Subset, ConcatDataset
import PIL
from utee import selector
import pdb
import numpy as np
from sklearn.model_selection import train_test_split

def get_dataloader(dataset, datadir, bs, augment=False, additional_idxs=None):
    if dataset == 'imnet':
        return get_imnet_dataloader(datadir, batch_size=bs, resize_dim=256) 
    elif dataset == 'cifar10':
        if 'val' in datadir:
            return get_cifar10_dataloader(train=False, batch_size=bs)
        return get_cifar10_dataloader(train=True, batch_size=bs)
    elif dataset == 'cifar100':
        if 'val' in datadir:
            return get_cifar100_dataloader(train=False, batch_size=bs)
        return get_cifar100_dataloader(train=True, batch_size=bs)
    elif dataset == 'stl10':
        if 'val' in datadir:
            return get_stl10_dataloader(train=False, batch_size=bs)
        return get_stl10_dataloader(train=True, batch_size=bs)
    elif dataset == 'svhn':
        if 'val' in datadir:
            return get_svhn_dataloader(train=False, batch_size=bs)
        return get_svhn_dataloader(train=True, batch_size=bs) 
    elif dataset == 'flowers102':
        if 'val' in datadir:
            return get_flowers_dataloader(train=False, augment=augment,batch_size=bs)
        return get_flowers_dataloader(train=True, augment=augment,batch_size=bs, 
                                      additional_idxs=additional_idxs)
    elif dataset == 'mnist':
        if 'val' in datadir:
            return get_mnist_dataloader(train=False, augment=augment, batch_size=bs)
        return get_mnist_dataloader(train=True, augment=augment, batch_size=bs)

def get_mnist_dataloader(train, augment, batch_size=32, pct=1): 
    dataset = datasets.MNIST('../data', train=train, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))]))
    dataloader = torch.utils.data.DataLoader(dataset,
                       batch_size=batch_size, shuffle=False)
    return  dataloader
    
def get_flowers_dataloader(train, augment, batch_size=32, additional_idxs=None):
    image_size = 256
    crop_size = 224
    normalize = transforms.Normalize(mean=[0.5208, 0.4205, 0.3441],
                                     std=[0.2944, 0.2465, 0.2735])
    
    shuffle=False
    if train and augment:
        data_path = './datasets/flowers102/train'
        d_transforms = transforms.Compose([
                transforms.RandomResizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,])
        shuffle=True
    else:
        data_path= './datasets/flowers102/test'
        d_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize
        ])
    # subselect for idxs if idxs are provided
    # This subselection is only used for the test_idxs
    dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=d_transforms)
    if train and len(additional_idxs):
        data_path = './datasets/flowers102/test'
        add_dataset = torchvision.datasets.ImageFolder(
            root=data_path,
            transform=d_transforms)
        add_dataset = Subset(add_dataset, additional_idxs)
        dataset = ConcatDataset([dataset, add_dataset])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=shuffle)
    return dataloader


def get_flowers_dataloader_pct(pct, batch_size=32): 
    image_size = 256
    crop_size = 224
    shuffle=True
    normalize = transforms.Normalize(mean=[0.5208, 0.4205, 0.3441],
                                     std=[0.2944, 0.2465, 0.2735])

    data_path = './datasets/flowers102/train'
    d_transforms = transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,])
    dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=d_transforms)
    
    # subselect for idxs if idxs are provided
    # This subselection is only used for the test_idxs
    dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=d_transforms)
    idxs = list(range(len(dataset)))
    train_idxs, _ = train_test_split(idxs, train_size=pct, stratify=dataset.targets)
    subset_dataset = Subset(dataset, train_idxs)
    dataloader = torch.utils.data.DataLoader(
        subset_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=shuffle)
    return dataloader

def get_cifar100_dataloader(train=False, batch_size=32):
    data_root='/tmp/public_dataset/pytorch'
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    dataset = datasets.CIFAR100(
                root=data_root, train=train, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                    transforms.Pad(4)),
                ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=False)
    return dataloader

def get_stl10_dataloader(train=False, batch_size=32):
    split = 'test'
    if train:
        split = 'train'
    data_root='/tmp/public_dataset/pytorch'
    data_root = os.path.expanduser(os.path.join(data_root, 'stl10-data'))
    dataset = datasets.STL10(root=data_root, split=split, download=True,
                             transform=transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=False)
    return dataloader

def get_imnet_dataloader(valdir, batch_size=12, resize_dim=256, normalize=True):
    n_workers = 1
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if normalize: 
        dataset = datasets.ImageFolder(valdir, transforms.Compose([
                    transforms.Resize(resize_dim, interpolation=PIL.Image.BICUBIC),
                    transforms.CenterCrop(256),
                    transforms.ToTensor(),
                    normalize,
                    ]))
    else:
        datasets.ImageFolder(valdir, transforms.Compose([
                    transforms.Resize(resize_dim, interpolation=PIL.Image.BICUBIC),
                    transforms.CenterCrop(256),
                    transforms.ToTensor(),
                ]))
    val_loader = torch.utils.data.DataLoader(dataset,
                batch_size=batch_size, shuffle=False,
                num_workers=n_workers, pin_memory=False)
    
    return val_loader
