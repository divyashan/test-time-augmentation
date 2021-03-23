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
from utils.imagenet_utils import accuracy
from utils.gpu_utils import restrict_GPU_pytorch
# Take in indices of diff augmentations
from expmt_vars import dataset, n_classes, tta_policy, results_dir, val_output_dir, aggregated_outputs_dir
from expmt_vars import aug_order, train_output_dir 

class ModelOutputs():
    def __init__(self, model_name, aug_name, dataset, mode):
        self.model_name = model_name 
        if mode == 'val':
            self.outputs_path= val_output_dir + '/'  + model_name + '_test.h5'
        else:
            self.outputs_path = val_output_dir+ '/' + model_name + '_val.h5'
        self.agg_outputs_folder = aggregated_outputs_dir + '/' + mode + '/' + model_name + '/' + aug_name + '/'  
        self.dataset = dataset

        self.aug_name = aug_name
        self.aug_idxs = get_aug_idxs(self.aug_name)
        
        with h5py.File(self.outputs_path) as hf:
            self.val_key_prefixes = self.process_keys(hf.keys())
        if not os.path.exists(self.agg_outputs_folder):
            os.makedirs(self.agg_outputs_folder)
                
    def apply(self, agg_name, n_runs=5):
        agg_f = get_agg_f(self.aug_name, agg_name, self.model_name, self.dataset, n_classes)
        agg_outputs_path = self.agg_outputs_folder + '/' + agg_name + '.h5'
        top1_runs, top5_runs = [[] for i in range(n_runs)], [[] for i in range(n_runs)]
        with h5py.File(agg_outputs_path, 'w') as hf_agg:
            with h5py.File(self.outputs_path) as hf:
                # Pre-computed aggregated outputs
                for key_pre in tqdm(self.val_key_prefixes):
                    if key_pre + '_outputs' in hf_agg.keys():
                        print("Using pre-computed values")
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
                    n_inputs = len(labels)
                    for i in range(n_runs):
                        idxs = np.random.choice(n_inputs, int(.8*n_inputs), replace=False)
                        scores= accuracy(agg_outputs[idxs], labels[idxs], topk=(1,5))
                        top1_runs[i].append(scores[0].item())
                        top5_runs[i].append(scores[1].item())
        top1s = [np.mean(x) for x in top1_runs]
        top5s = [np.mean(x) for x in top5_runs]
        return top1s, top5s 

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

def evaluate_aggregation_outputs(model_name, dataset, mode='val'):
    # aug_names = ['hflip', 'orig','five_crop', 'colorjitter', 'rotation', 'combo']
    aug_names = ['orig', 'combo']
    agg_names = [ 'mean', 'gps', 'partial_lr', 'full_lr', 'ours', 'max'] 
 #   agg_names = [ 'partial_lr', 'full_lr', 'ours', 'max'] 
#    agg_names = ['mean']
    n_runs = 5 
    results = []
    for aug_name in aug_names:
        for agg_name in agg_names:
            print("AUG: ", aug_name, "\tAGG: ", agg_name)
            # Skip runs for aggregation over original prediction
            if aug_name == 'orig' and agg_name != 'mean':
                continue
            mo = ModelOutputs(model_name, aug_name, dataset, mode)
            # Combines + scores these model outputs
            top1s, top5s = mo.apply(agg_name)
            for i in range(n_runs):
                results.append({'model':model_name, 'aug':aug_name, 'agg':agg_name, 
                                'top1':top1s[i], 'top5':top5s[i], 'run': i})
            pd.DataFrame(results).to_csv(results_dir +  '/' + mode + '/' + model_name + '_agg_fs')

if __name__ == '__main__':

    if len(sys.argv) == 1:
        model_names = ['resnet18', 'resnet50', 'resnet101', 'MobileNetV2']
        for model_name in model_names:
            evaluate_aggregation_outputs(model_name)
    else:
        model_name = sys.argv[1]
        evaluate_aggregation_outputs(sys.argv[1])

