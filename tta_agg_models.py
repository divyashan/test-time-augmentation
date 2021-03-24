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
from models import get_pretrained_model
from scipy.special import softmax
from expmt_vars import n_classes
from scipy.optimize import nnls

class TTARegression(nn.Module):
    def __init__(self, n_augs, n_classes, temp_scale=1, initialization='even'):
        super().__init__()
        
        if initialization == 'even':
            self.coeffs = nn.Parameter(torch.randn((n_augs, n_classes), requires_grad=True, dtype=torch.float))
            self.coeffs.data.fill_(1.0/n_augs) 
        else:
            coeffs = torch.cat([torch.Tensor(initialization) for i in range(n_classes)], axis=1)
            self.coeffs = nn.Parameter(coeffs, requires_grad = True)

        self.temperature = temp_scale
    
    def forward(self, x):
        # Computes the outputs / predictions
        x = x/self.temperature
        mult = self.coeffs * x
        return mult.sum(axis=1)

class TTAPartialRegression(nn.Module):
    def __init__(self, n_augs, n_classes, temp_scale=1, initialization='even',coeffs=[]):
        super().__init__()
        # To make "a" and "b" real parameters of the model, we need to wrap them with nn.Parameter
        self.coeffs = nn.Parameter(torch.randn((n_augs,1 ), requires_grad=True, dtype=torch.float))
        self.temperature = temp_scale
        if len(coeffs):
            self.coeffs = nn.Parameter(torch.Tensor(coeffs), requires_grad=True) 
        else:
            if initialization == 'even':
                self.coeffs.data.fill_(1.0/n_augs) 
            elif initialization== 'original':
                self.coeffs.data[0,:].fill_(1)
                self.coeffs.data[1,:].fill_(0)
    
    def forward(self, x):
        # Computes the outputs / predictions
        x = x/self.temperature
        mult = torch.matmul(x.transpose(1, 2), self.coeffs / torch.sum(self.coeffs, axis=0))
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
            for j in range(len(remaining_idxs)):
                possible_outputs = new_weight*aug_outputs[j] + old_weight* current_preds
                nll_vals.append(log_loss(labels, possible_outputs, labels=np.arange(n_classes)))
            next_idx = remaining_idxs[np.argmin(nll_vals)]
            remaining_idxs.remove(next_idx)
            idxs.append(next_idx)
            current_preds = new_weight*outputs[next_idx] + old_weight*current_preds
        return idxs

    def forward(self, x):
        # Performing prediciton on the mean of the scaled versions would be the same
        mult = torch.mean(x[:,self.idxs], axis=1)
        return mult.squeeze()


# Predicting weights directly from image
# Results in Supplementary Material
class ImageWeights(nn.Module): 
    def __init__(self, model_name, n_augs, n_classes, n_features, orig_idx, dataset):
        super().__init__()
        self.model = get_pretrained_model(model_name, dataset)
        for param in self.model.parameters():
            param.requires_grad = False
        self.n_augs = n_augs
        self.orig_idx = orig_idx
        n_full = n_classes
        self.fc3 = nn.Linear(n_features, n_full)
        self.fc4 = nn.Linear(n_full, n_augs)
        self.sigmoid = nn.Sigmoid()
        self.sm = nn.Softmax(dim=1)
        
        nn.init.xavier_uniform(self.fc3.weight)
        nn.init.xavier_uniform(self.fc4.weight)

                
    def forward(self, x):
        # x is a [B, A, H, W] matrix 
        orig_image = x[self.orig_idx]
        f = self.model.features(orig_image)
        w = self.fc3(torch.flatten(f, 1))
        w = F.relu(w)
        w = self.fc4(w)
        w = self.sm(w)
        
        aug_preds = []
        for i in range(len(x)):
            aug_preds.append(self.model(x[i]))
        
        aug_preds = torch.stack(aug_preds)
        aug_preds = aug_preds.permute(2, 1, 0)
        aug_preds = aug_preds * w
        aug_pred = aug_preds.mean(axis=2)
        return aug_pred.permute(1, 0)
    
    def get_w(self, x):
        orig_image = x[self.orig_idx]
        f = self.model.features(orig_image)

        w = self.fc3(f)
        return  self.sm(w)
        
# Predicting when to choose TTA prediction over original model from image
# Results in Supplementary Material
class ImageDeferral(nn.Module):
    def __init__(self, model_name, n_classes, n_features, orig_idx, dataset):
        super().__init__()
        self.model = get_pretrained_model(model_name, dataset)
        for param in self.model.parameters():
            param.requires_grad = False
        self.orig_idx = orig_idx
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.dropout1 = nn.Dropout2d(0.25)
#         self.dropout2 = nn.Dropout2d(0.5) # why is the later one higher?
#         self.fc1 = nn.Linear(9216, 128)
        n_full = n_classes
        self.dropout1 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(n_features, n_full)
        self.fc4 = nn.Linear(n_full, 1)
        self.sigmoid = nn.Sigmoid()
        nn.init.xavier_uniform(self.fc2.weight)
        nn.init.xavier_uniform(self.fc4.weight)
        
    def forward(self, x):
        orig_image = x[self.orig_idx]
        f = self.model.features(orig_image)
        presig_s = self.fc2(torch.flatten(f, 1))
        presig_s = F.relu(presig_s)
        presig_s = self.fc4(presig_s)
        s = self.sigmoid(presig_s)
        
        aug_preds = []
        for i in range(len(x)):
            aug_preds.append(self.model(x[i]))
        
        aug_preds = torch.stack(aug_preds)
        aug_preds = aug_preds.permute(1, 0, 2)
        aug_pred = aug_preds.mean(axis=1)
        orig_pred = self.model(x[self.orig_idx])
        
        return (1-s)*orig_pred + (s)*aug_pred
    
    def get_s(self, x): 
        f = self.model.features(x[self.orig_idx])
        presig_s = self.fc2(torch.flatten(f, 1))
        presig_s = F.relu(presig_s)
        presig_s = self.dropout1(presig_s)
        presig_s = self.fc4(presig_s)
        s = self.sigmoid(presig_s)
        return s
