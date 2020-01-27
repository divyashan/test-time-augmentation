import os
import torch
import numpy as np
from augmentations import get_aug_idxs
from tta_agg_models import TTARegression
from tta_train import train_tta_lr 

def get_agg_f(aug_name, agg_name, model_name):
    aug_idxs = get_aug_idxs(aug_name)
    if agg_name == 'mean':
        return mean_agg_f(len(aug_idxs))
    elif agg_name == 'lr':
        model_path = './agg_models/'+model_name+'/'+aug_name + '/lr.pth'
        if not os.path.exists(model_path):
            print("[ ] Training LR model")
            # train the model
            model = train_tta_lr(model_name, aug_name, 5) 
        print("[X] LR Model Trained") 
        model = TTARegression(len(aug_idxs),1000,'even')
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    # could do max too?
    # add support for naming the learned parameters
    # lr_agg_f(aug_idxs, model_name)

def mean_agg_f(n_augs):#
    coeffs = np.zeros((n_augs, 1000))
    coeffs[:,:] = 1/n_augs
    coeffs = torch.Tensor(coeffs)
    def agg_f(inputs):
        mult = inputs*coeffs
        return mult.sum(axis=1)
    return agg_f
