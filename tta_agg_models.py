from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import h5py
import numpy as np
import time
import pdb 
import itertools 

import torch
from torch import nn, optim
from torch.nn.functional import softmax as torch_softmax
from scipy.special import softmax
from expmt_vars import n_classes

class TTARegression(nn.Module):
    def __init__(self, n_augs, n_classes, temp_scale=1, initialization='even'):
        super().__init__()
        # To make "a" and "b" real parameters of the model, we need to wrap them with nn.Parameter
        self.coeffs = nn.Parameter(torch.randn((n_augs, n_classes), requires_grad=True, dtype=torch.float))
        self.temperature = temp_scale
        if initialization == 'even':
            self.coeffs.data.fill_(1.0/n_augs) 
        elif initialization== 'original':
            self.coeffs.data[0,:].fill_(1)
            self.coeffs.data[1,:].fill_(0)
    
    def forward(self, x):
        # Computes the outputs / predictions
        x = x/self.temperature
        mult = self.coeffs * x
        return mult.sum(axis=1)

class TTAPartialRegression(nn.Module):
    def __init__(self, n_augs, n_classes, temp_scale=1, initialization='even'):
        super().__init__()
        # To make "a" and "b" real parameters of the model, we need to wrap them with nn.Parameter
        self.coeffs = nn.Parameter(torch.randn((n_augs,1 ), requires_grad=True, dtype=torch.float))
        self.temperature = temp_scale
        if initialization == 'even':
            self.coeffs.data.fill_(1.0/n_augs) 
        elif initialization== 'original':
            self.coeffs.data[0,:].fill_(1)
            self.coeffs.data[1,:].fill_(0)
             
    def forward(self, x):
        # Computes the outputs / predictions
        x = x/self.temperature
        mult = torch.matmul(x.transpose(1, 2), self.coeffs)
        return mult.squeeze()

class GPS(nn.Module):
    def __init__(self, n_subpolicies, train_path, scale):
        super().__init__()
        self.scale = scale 
        self.n_subpolicies = n_subpolicies
        self.idxs = self.get_idxs(train_path)
    
    def get_idxs(self, train_path):
        # Implementating of GPS pap 
        # Get outputs

        hf = h5py.File(train_path, 'r')
        outputs = hf['batch1_inputs'][:] 
        labels = hf['batch1_labels'][:]
         
        outputs = outputs / self.scale 
        n_augs, n_examples, n_classes = outputs.shape
        remaining_idxs = list(np.arange(outputs.shape[0]))
        softmaxed = [np.expand_dims(softmax(aug_o, axis=1), axis=0) for aug_o in outputs]
        outputs = np.concatenate(softmaxed, axis=0)
        idxs = []
        current_preds = np.zeros((n_examples, n_classes))
        for i in range(self.n_subpolicies):
            aug_outputs = outputs[remaining_idxs]
            # should be of shape number of remaining augs, 
            old_weight = i/self.n_subpolicies
            new_weight = 1-old_weight
            
            # calculate NLL for each possible output
            nll_vals = []
            for i in range(len(remaining_idxs)):
                possible_outputs = new_weight*aug_outputs[i] + old_weight* current_preds
                try:
                    nll_vals.append(log_loss(labels, possible_outputs, labels=np.arange(n_classes)))
                except:
                    pdb.set_trace()
            next_idx = remaining_idxs[np.argmin(nll_vals)]
            remaining_idxs.remove(next_idx)
            idxs.append(next_idx)
        print('IDXS: ', idxs)
        return idxs

    def forward(self, x):
        # Performing prediciton on the mean of the scaled versions would be the same
        mult = torch.mean(x[:,self.idxs], axis=1)
        return mult.squeeze()

