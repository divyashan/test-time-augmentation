import torch
import torch.nn.functional as F
import torchvision.transforms.functional as FV
from torchvision import transforms
import pdb
import PIL

import albumentations as A

def jitter(img, brightness, contrast, saturation, hue):
    jittered = torch.stack([jitter_image(img[i], brightness, contrast, saturation, hue) 
                            for i in range(img.shape[0])])
    return jittered

def jitter_image(single_img, brightness, contrast, saturation, hue):
    transform = A.RandomBrightness(limit=.1, p=1) 
    jj = transform(image=single_img.cpu().numpy())['image']
    jj = torch.Tensor(jj)
    jj = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])(jj)
    return jj

def rotate(imgs, angle):
    rotated= torch.stack([rotate_img_range(imgs[i], angle) for i in range(imgs.shape[0])])
    return rotated 
    
def rotate_img_range(single_img, angle):
    transform = A.Rotate(limit=angle)
    jj = transform(image=single_img.cpu().numpy())['image']
    jj =  torch.Tensor(jj)
    return jj 

def rotate_img(single_img, angle):

    jj = transforms.ToPILImage()(single_img.cpu())
    jj = jj.rotate(angle, resample=PIL.Image.NEAREST)
    jj = transforms.ToTensor()(jj)
    return jj 


def rot90(x, k=1):
    """rotate batch of images by 90 degrees k times"""
    return torch.rot90(x, k, (2, 3))


def hflip(x):
    """flip batch of images horizontally"""
    return x.flip(3)


def vflip(x):
    """flip batch of images vertically"""
    return x.flip(2)


def sum(x1, x2):
    """sum of two tensors"""
    return x1 + x2


def add(x, value):
    """add value to tensor"""
    return x + value


def max(x1, x2):
    """compare 2 tensors and take max values"""
    return torch.max(x1, x2)


def min(x1, x2):
    """compare 2 tensors and take min values"""
    return torch.min(x1, x2)


def multiply(x, factor):
    """multiply tensor by factor"""
    return x * factor


def scale(x, scale_factor, interpolation="nearest", align_corners=None):
    """scale batch of images by `scale_factor` with given interpolation mode"""
    h, w = x.shape[2:]
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    scaled_up =  F.interpolate(
        x, size=(new_h, new_w), mode=interpolation, align_corners=align_corners
    )
    return center_crop(scaled_up, h, w)


def resize(x, size, interpolation="nearest", align_corners=None):
    """resize batch of images to given spatial size with given interpolation mode"""
    return F.interpolate(x, size=size, mode=interpolation, align_corners=align_corners)


def crop(x, x_min=None, x_max=None, y_min=None, y_max=None):
    """perform crop on batch of images"""
    return x[:, :, y_min:y_max, x_min:x_max]


def crop_lt(x, crop_h, crop_w):
    """crop left top corner"""
    orig_h = x.shape[2]
    orig_w = x.shape[3]
    x = x[:, :, 0:crop_h, 0:crop_w]
    return x
    #return resize(x, (orig_h, orig_w))

def crop_lb(x, crop_h, crop_w):
    """crop left bottom corner"""
    orig_h = x.shape[2]
    orig_w = x.shape[3]
    x = x[:, :, -crop_h:, 0:crop_w]
    return x
    #return resize(x, (orig_h, orig_w))

def crop_rt(x, crop_h, crop_w):
    """crop right top corner"""
    orig_h = x.shape[2]
    orig_w = x.shape[3]
    x = x[:, :, 0:crop_h, -crop_w:]
    return x
    #return resize(x, (orig_h, orig_w))

def crop_rb(x, crop_h, crop_w):
    """crop right bottom corner"""
    orig_h = x.shape[2]
    orig_w = x.shape[3]
    x = x[:, :, -crop_h:, -crop_w:]
    return x
    #return resize(x, (orig_h, orig_w))

def crop_c(x, crop_h, crop_w):
    orig_h = x.shape[2]
    orig_w = x.shape[3]
    x = center_crop(x, crop_h, crop_w)
    #return resize(x, (orig_h, orig_w))
    return x

def crop_orig(x, crop_h, crop_w):
    return x

def center_crop(x, crop_h, crop_w):
    """make center crop"""
    orig_h = x.shape[2]
    orig_w = x.shape[3]

    center_h = x.shape[2] // 2
    center_w = x.shape[3] // 2
    half_crop_h = crop_h // 2
    half_crop_w = crop_w // 2

    y_min = center_h - half_crop_h
    y_max = center_h + half_crop_h + crop_h % 2
    x_min = center_w - half_crop_w
    x_max = center_w + half_crop_w + crop_w % 2

    return x[:, :, y_min:y_max, x_min:x_max]
