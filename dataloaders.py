import os
import torch
import torchvision
from torchvision import transforms
from torchvision import datasets
import PIL
from utee import selector
import pdb

def get_dataloader(dataset, datadir, bs, augment=False):
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
        return get_flowers_dataloader(train=True, augment=augment,batch_size=bs)
    elif dataset == 'birds200':
        if 'val' in datadir:
            return get_birds_dataloader(train=False, augment=augment,batch_size=bs)
        return get_birds_dataloader(train=True, augment=augment, batch_size=bs)
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
    
def get_flowers_dataloader(train, augment, batch_size=32):
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
    
    dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=d_transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=shuffle)
    return dataloader

def get_birds_dataloader(train, augment, batch_size=32):
    image_size = 256
    crop_size = 224
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    
    shuffle=False
    if train and augment:
        data_path = './datasets/cub200/train'
        d_transforms = transforms.Compose([
                transforms.RandomResizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,])
        shuffle=True
    else:
        data_path= './datasets/cub200/test'
        d_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize
        ])
    
    dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=d_transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=shuffle)
    return dataloader

def get_svhn_dataloader(train=False, batch_size=32):
    split = 'test'
    if train:
        split = 'train'
    data_root='/tmp/public_dataset/pytorch'
    data_root = os.path.expanduser(os.path.join(data_root, 'svhn-data'))
    dataloader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root=data_root, split=split, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
            ),
            batch_size=batch_size, shuffle=False, pin_memory=False)
    return dataloader

def get_cifar10_dataloader(train=False, batch_size=32):
    data_root='/tmp/public_dataset/pytorch'
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    dataset = datasets.CIFAR10(
                root=data_root, train=train, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=False) 
    return dataloader

def get_cifar100_dataloader(train=False, batch_size=32):
    data_root='/tmp/public_dataset/pytorch'
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    dataset = datasets.CIFAR100(
                root=data_root, train=train, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=False)
    return dataloader

def get_imnet_dataloader(valdir, batch_size=12, resize_dim=256):
    n_workers = 8
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(resize_dim, interpolation=PIL.Image.BICUBIC),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=n_workers, pin_memory=False)
    
    return val_loader
