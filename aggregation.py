import os
import torch
import numpy as np
from augmentations import get_aug_idxs
from tta_agg_models import TTARegression, TTAPartialRegression, GPS, ImprovedLR
from tta_train import train_tta_lr 
from expmt_vars import agg_models_dir, val_output_dir
import pdb

from utils.tta_utils import get_calibration

def get_agg_f(aug_name, agg_name, model_name, dataset, n_classes):
    aug_idxs = get_aug_idxs(aug_name)
    orig_idx = get_aug_idxs('orig')
    val_path = val_output_dir + '/' + model_name + '_val.h5'
    temp_scale = get_calibration(val_path, orig_idx)
    #temp_scale = 1
    if agg_name == 'mean':
        return mean_agg_f(len(aug_idxs), n_classes)
    elif agg_name == 'improved_lr':
        model_path = agg_models_dir + '/'+model_name+'/'+aug_name + '/partial_lr.pth'
        model = TTAPartialRegression(len(aug_idxs),n_classes,temp_scale,'even')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
        coeffs = model.coeffs
        model = ImprovedLR(len(aug_idxs), n_classes,temp_scale, partial_lr_init=coeffs)
        # TOOD: Add loading from saved model
        model.fit(val_path)
        pdb.set_trace()
        return model
    elif agg_name == 'full_lr':
        n_epochs = 30 
        model_path = agg_models_dir + '/'+model_name+'/'+aug_name + '/full_lr.pth'
        if not os.path.exists(model_path):
            print("[ ] Training LR model")
            # train the model
            model = train_tta_lr(model_name, aug_name, n_epochs, 'full', dataset, n_classes, 
                                 temp_scale) 
        print("[X] Full LR Model Trained") 
        model = TTARegression(len(aug_idxs),n_classes,temp_scale, 'even')
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    elif agg_name == 'partial_lr':
        n_epochs = 15 
        model_path = agg_models_dir + '/'+model_name+'/'+aug_name + '/partial_lr.pth'
        if not os.path.exists(model_path):
            print("[ ] Training LR model")
            # train the model
            model = train_tta_lr(model_name, aug_name, n_epochs, 'partial', dataset, n_classes,
                                 temp_scale) 
        print("[X] Partial LR Model Trained") 
        model = TTAPartialRegression(len(aug_idxs),n_classes,temp_scale,'even')
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    elif agg_name == 'max':
        return max_agg_f()        
    elif agg_name == 'gps':
        temp_scale = get_calibration(val_path, orig_idx)
        n_policies = 3
        model = GPS(n_policies, val_path, temp_scale)
        return model
        
    # could do max too?
    # add support for naming the learned parameters
    # lr_agg_f(aug_idxs, model_name)

def mean_agg_f(n_augs, n_classes):#
    coeffs = np.zeros((n_augs, n_classes))
    coeffs[:,:] = 1/n_augs
    coeffs = torch.Tensor(coeffs)
    def agg_f(inputs):
        mult = inputs*coeffs
        return mult.sum(axis=1)
    return agg_f

def max_agg_f():
    def agg_f(inputs):
        n_examples = inputs.shape[0]
        return inputs[np.arange(n_examples),inputs.max(axis=2).values.max(axis=1).indices,:]
    return agg_f
