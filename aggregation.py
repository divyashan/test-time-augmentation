import os
import torch
import numpy as np
from augmentations import get_aug_idxs
from tta_agg_models import TTARegression, TTAPartialRegression, GPS
from tta_train import train_tta_lr
from expmt_vars import agg_models_dir, val_output_dir
from utils.imagenet_utils import accuracy
import pdb
import h5py

from utils.tta_utils import get_calibration

def get_agg_f(aug_name, agg_name, model_name, dataset, n_classes):
    aug_idxs = get_aug_idxs(aug_name)
    orig_idx = get_aug_idxs('orig')
    n_augs = len(aug_idxs)
    val_path = val_output_dir + '/' + model_name + '_val.h5'
    temp_scale = get_calibration(val_path, orig_idx)
    agg_file_path = agg_models_dir + '/' + model_name + '/' + aug_name 
    if not os.path.exists(agg_file_path):
        os.makedirs(agg_file_path)
    #temp_scale = 1
    if agg_name == 'mean':
        return mean_agg_f(len(aug_idxs), n_classes)
    elif agg_name == 'full_lr':
        n_epochs = 30 
        coeffs = 'even' 
        model_path = agg_models_dir + '/'+model_name+'/'+aug_name + '/full_lr.pth'
        if not os.path.exists(model_path):
            model = train_tta_lr(model_name, aug_name, n_epochs, 
                                 'full', dataset, n_classes, 1, coeffs) 
        model = TTARegression(n_augs ,n_classes, 1, 'even')
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    
    elif agg_name == 'partial_lr':
        n_epochs = 20 
        model_path = agg_models_dir + '/'+model_name+'/'+aug_name + '/partial_lr.pth'
        if not os.path.exists(model_path):
            model = train_tta_lr(model_name, aug_name, n_epochs, 'partial', dataset, n_classes,
                                 1) 
        model = TTAPartialRegression(n_augs, n_classes, 1, 'even')
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    
    elif agg_name == 'ours':
        plr_path = agg_models_dir + '/'+model_name+'/'+aug_name + '/partial_lr.pth'
        flr_path = agg_models_dir + '/'+model_name+'/'+aug_name + '/full_lr.pth'
        flr_model = TTARegression(n_augs,n_classes, 1, 'even')
        flr_model.load_state_dict(torch.load(flr_path))

        plr_model = TTAPartialRegression(n_augs,n_classes,temp_scale,'even')
        plr_model.load_state_dict(torch.load(plr_path))

        val_f = h5py.File(val_output_dir + '/' + model_name + '_val_val.h5', 'r')
        outputs = val_f['batch1_inputs']
        labels = val_f['batch1_labels']

        outputs = torch.Tensor(np.swapaxes(outputs, 0, 1))
        plr_preds = plr_model(outputs).detach().cpu().numpy()
        flr_preds = flr_model(outputs).detach().cpu().numpy()

        flr_acc = accuracy(torch.Tensor(flr_preds), torch.Tensor(labels[:]), topk=(1, 5))[0].item()
        plr_acc = accuracy(torch.Tensor(plr_preds), torch.Tensor(labels[:]), topk=(1, 5))[0].item()
        # evaluate accuracy of this on val val 
        # return model with higher accuracy
        if flr_acc > plr_acc:
            return flr_model
        return plr_model

    elif agg_name == 'max':
        return max_agg_f()        
    
    elif agg_name == 'gps':
        temp_scale = get_calibration(val_path, orig_idx)
        n_policies = 3
        model = GPS(n_policies, val_path, temp_scale)
        return model
    
    elif agg_name == 'image_deferral':
        model = ImageDeferral(model_name, n_augs, n_classes, n_features, orig_idx, dataset)
    elif agg_name == 'image_weights':
        model = ImageWeights(model_name, n_augs, n_classes, n_features, orig_idx, dataset)
        
        
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
