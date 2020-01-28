import os
import pdb
import sys
import h5py
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm 
from augmentations import get_aug_idxs
from aggregation import get_agg_f
from imagenet_utils import accuracy
from gpu_utils import restrict_GPU_pytorch
# Take in indices of diff augmentations
class ModelOutputs():
    def __init__(self, model_name, aug_name):
        self.model_name = model_name 
        self.val_outputs_path= './outputs/model_outputs/val/' + model_name + '.h5'
        self.agg_outputs_folder = './outputs/aggregated_outputs/' + model_name + '/' + aug_name + '/'  
       
        self.n_total_augs = 60
        self.aug_name = aug_name
        self.aug_idxs = get_aug_idxs(self.aug_name)
        
        with h5py.File(self.val_outputs_path) as hf:
            self.val_key_prefixes = self.process_keys(hf.keys())
        if not os.path.exists(self.agg_outputs_folder):
            os.makedirs(self.agg_outputs_folder)
                
    def apply(self, agg_name):
        agg_f = get_agg_f(self.aug_name, agg_name, self.model_name)
        agg_outputs_path = self.agg_outputs_folder + '/' + agg_name + '_test.h5'
        top1s, top5s = [], []
        with h5py.File(agg_outputs_path) as hf_agg:
            with h5py.File(self.val_outputs_path) as hf:
                # Pre-computed aggregated outputs
                for key_pre in tqdm(self.val_key_prefixes):
                    if key_pre + '_outputs' in hf_agg.keys():
                        agg_outputs = hf_agg[key_pre + '_outputs'].value
                        labels = hf_agg[key_pre + '_labels'].value
                    else:
                        outputs = hf[key_pre+'_inputs'].value[self.aug_idxs,:]
                        labels= hf[key_pre+'_labels'].value
                        agg_outputs, labels = self.batch_apply(agg_f, outputs, labels)
                        hf_agg.create_dataset(key_pre + '_outputs', data=agg_outputs)
                        hf_agg.create_dataset(key_pre + '_labels', data=labels)
                    agg_outputs = torch.Tensor(agg_outputs)
                    labels = torch.Tensor(labels)
                    scores= accuracy(agg_outputs, labels, topk=(1,5))
                    print(len(agg_outputs))
                    top1s.append(scores[0].item())
                    top5s.append(scores[1].item())
        return np.mean(top1s), np.mean(top5s)

    def batch_apply(self, agg_f, outputs, labels):
        with torch.no_grad():
            outputs = np.swapaxes(outputs, 0, 1)
            outputs = torch.Tensor(outputs)
            agg_outputs = agg_f(outputs)
            agg_outputs = torch.Tensor(agg_outputs)
            labels = torch.Tensor(labels)
        return agg_outputs, labels
        # TODO: Write out aggregated outputs to model/aug_name/agg.h5
        
    def process_keys(self, key_list):
        keys = ['_'.join(x.split('_')[:-1]) for x in key_list]
        return keys

def evaluate(model_name, aug_name, agg_name):
    # Gets relevant model outputs to augmentation
    mo = ModelOutputs(model_name, aug_name)
    # Combines + scores these model outputs 
    score = mo.apply(agg_name)
    return score

def write_aggregation_outputs(model_name):
    aug_names = ['hflip', 'orig','five_crop', 'colorjitter', 'rotation', 'combo']
    agg_names = ['mean', 'partial_lr', 'full_lr', 'max'] 
    
    results = []
    for aug_name in aug_names:
        for agg_name in agg_names:
            print("AUG: ", aug_name, "\tAGG: ", agg_name)
            if aug_name == 'orig' and agg_name != 'mean':
                continue
            mo = ModelOutputs(model_name, aug_name)
            # Combines + scores these model outputs
            top1, top5 = mo.apply(agg_name)
            results.append({'model':model_name, 'aug':aug_name, 'agg':agg_name, 'top1':top1, 'top5':top5})
            pd.DataFrame(results).to_csv('./results/' + model_name + '_agg_fs')

if __name__ == '__main__':
    model_name = sys.argv[1]
    write_aggregation_outputs(sys.argv[1])
    #print('Score: ', evaluate(model_name, 'five_crop', 'lr'))
