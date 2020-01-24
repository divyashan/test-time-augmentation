from sklearn.linear_model import LogisticRegression
import h5py
import numpy as np
import time
import pdb 
import itertools 

import torch
from torch import nn

from gpu_utils import restrict_GPU_pytorch
from imagenet_utils import accuracy, AverageMeter, ProgressMeter


class TTARegression(nn.Module):
    def __init__(self, n_augs, n_classes, initialization='even'):
        super().__init__()
        # To make "a" and "b" real parameters of the model, we need to wrap them with nn.Parameter
        self.coeffs = nn.Parameter(torch.randn((n_augs, n_classes), requires_grad=True, dtype=torch.float))
        if initialization == 'even':
            self.coeffs.data.fill_(1.0/n_augs) 
        elif initialization== 'original':
            self.coeffs.data[0,:].fill_(1)
            self.coeffs.data[1,:].fill_(0)
             
    def forward(self, x):
        # Computes the outputs / predictions
        mult = self.coeffs * x
        return mult.sum(axis=1)

