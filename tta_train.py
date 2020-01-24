import os
import pdb
import torch
import h5py
import numpy as np
from tqdm import tqdm

from tta_agg_models import TTARegression
from imagenet_utils import accuracy, AverageMeter, ProgressMeter
from augmentations import get_aug_idxs

def train_tta_lr(model_name, aug_name, epochs):
    idxs = get_aug_idxs(aug_name)
    datapath = './outputs/model_outputs/train/' + model_name + '.h5'
    
    epochs = 10
    n_augs = len(idxs)
    criterion = torch.nn.CrossEntropyLoss()
    model = TTARegression(n_augs,1000,'even')
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=1e-4)

    model.cuda('cuda:0')
    criterion.cuda('cuda:0')
    model.train()
    
    with h5py.File(datapath) as hf:
        for epoch in range(epochs):
            for key in tqdm(hf.keys()):
                if 'labels' in key:
                    continue
                examples = np.swapaxes(hf[key][idxs,:], 0, 1)
                target = hf[key[:-7] +'_labels']
                
                examples = torch.Tensor(examples)
                target = torch.Tensor(target).long()
                examples = examples.cuda('cuda:0', non_blocking=True)
                target = target.cuda('cuda:0', non_blocking=True)
         
                output = model(examples)
                loss = criterion(output, target)
                acc1, acc5 = accuracy(output, target, topk=(1,5))
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()   
            model_prefix = './agg_models/' + model_name + '/' + aug_name 
            if not os.path.exists(model_prefix):
                os.makedirs(model_prefix)
            torch.save(model.state_dict(), model_prefix + '/lr.pth')
    return model
