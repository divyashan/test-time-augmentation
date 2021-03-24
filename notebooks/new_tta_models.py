from torch import nn
import torch
import pdb
import torch.nn.functional as F
import numpy as np

class ImageW(nn.Module):
    
    def __init__(self, model, n_augs, n_classes, n_features, orig_idx):
        super().__init__()
        self.model = model
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
        

class ImageS(nn.Module):
    
    def __init__(self, model, n_classes, n_features, orig_idx):
        super().__init__()
        self.model = model
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
        # x is a [B, A, H, W] matrix 
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

class ImageWS(nn.Module):
    
    def __init__(self, model, n_augs, orig_idx, n_features):
        super().__init__()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        self.n_augs = n_augs
        self.orig_idx = orig_idx
        self.fc2 = nn.Linear(n_features, 1)
        self.fc3 = nn.Linear(n_features, n_augs)
        self.sigmoid = nn.Sigmoid()
        self.sm = nn.Softmax(dim=1)
                
    def forward(self, x):
        # x is a [B, A, H, W] matrix 
        orig_image = x[self.orig_idx]
        f = self.model.features(orig_image)
        w = self.fc3(torch.flatten(f, 1))
        presig_s = self.fc2(torch.flatten(f, 1))
        
        s = self.sigmoid(presig_s)
        w = self.sm(w)
        
        aug_preds = []
        for i in range(len(x)):
            aug_preds.append(self.model(x[i]))
        
        aug_preds = torch.stack(aug_preds)
        aug_preds = aug_preds.permute(2, 1, 0)
        aug_preds = aug_preds * w
        aug_pred = aug_preds.mean(axis=2)
        aug_pred = aug_pred.permute(1, 0)
        orig_pred = self.model(x[self.orig_idx])
        
        return (1-s)*orig_pred + (s)*aug_pred
        
    def get_sw(self, x): 
        f = self.model.features(x[self.orig_idx])
        w = self.fc3(torch.flatten(f, 1))
        presig_s = self.fc2(torch.flatten(f, 1))
        
        s = self.sigmoid(presig_s)
        w = self.sm(w)
        return s, w

class AugWeights(nn.Module):
    
    def __init__(self, model, n_augs):
        super().__init__()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        self.n_augs = n_augs
        self.w = nn.Parameter(torch.randn((n_augs, 1), requires_grad=True, dtype=torch.float))
        self.w.data.fill_(1.0/n_augs)
        self.sm = nn.Softmax(dim=0)
                 
    def forward(self, x):
        aug_preds = []
        for i in range(len(x)):
            aug_preds.append(self.model(x[i]))
        
        aug_preds = torch.stack(aug_preds)
        aug_preds = aug_preds.permute(1, 0, 2)
        
        #pos_w = self.w - self.w.min(0, keepdim=True)[0]
        #norm_w = pos_w / pos_w.max(0, keepdim=True)[0]
        #norm_w = self.w + self.w.min() 
        #norm_w = norm_w/ norm_w.max()
        aug_pred = aug_preds * self.w
        aug_pred = aug_pred.mean(axis=1)
        
        return aug_pred
    
class ClassWeights(nn.Module):
    
    def __init__(self, model, n_augs, n_classes):
        super().__init__()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        self.n_augs = n_augs
        self.w = nn.Parameter(torch.randn((n_augs, n_classes), requires_grad=True, dtype=torch.float))
        self.w.data.fill_(1.0/n_augs)
        self.sm = nn.Softmax(dim=0)
                 
    def forward(self, x):
        aug_preds = []
        for i in range(len(x)):
            aug_preds.append(self.model(x[i]))
        
        aug_preds = torch.stack(aug_preds)
        aug_preds = aug_preds.permute(1, 0, 2)
        
        #pos_w = self.w - self.w.min(0, keepdim=True)[0]
        #norm_w = pos_w / pos_w.max(0, keepdim=True)[0]
        #norm_w = self.w + self.w.min() 
        #norm_w = norm_w/ norm_w.max()
        aug_pred = aug_preds * self.w
        #aug_pred = aug_preds * self.w
        aug_pred = aug_pred.mean(axis=1)
        
        return aug_pred
class StandardTTA(nn.Module):
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        aug_preds = []
        for i in range(len(x)):
            aug_preds.append(self.model(x[i]))

        aug_preds = torch.stack(aug_preds)
        aug_preds = aug_preds.permute(1, 0, 2)
        final_pred = torch.mean(aug_preds, 1)
        return final_pred

class Original(nn.Module):
    def __init__(self, model, orig_idx):
        super().__init__()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        self.orig_idx = orig_idx
        
    def forward(self, x):
        orig_pred = self.model(x[self.orig_idx])
        return orig_pred
        
