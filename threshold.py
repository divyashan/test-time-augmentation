# Given a model_name + aug_name, apply thresholding to the aggregated outputs
# Where thresholding is using the original prediction depending on the predicted probability
import pdb
import sys
import h5py
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.special import softmax

from utils.gpu_utils import restrict_GPU_pytorch
from utils.imagenet_utils import accuracy
from aug_cartesian_product import AUG_ORDER

def evaluate_threshold(model_name):
    aug_names = AUG_ORDER + ['combo']
    agg_name = 'partial_lr'
    results = []

    for aug_name in aug_names:
        thresh, orig, aug = thresholding_acc(model_name, aug_name, agg_name) 
        results.append({'top1': thresh[0], 'top5': thresh[1], 'aug': aug_name, 'model': model_name, 'mode': 'thresh', 'agg': agg_name})
        results.append({'top1': orig[0], 'top5': orig[1], 'aug': aug_name, 'model': model_name, 'mode': 'orig', 'agg': agg_name})
        results.append({'top1': aug[0], 'top5': aug[1], 'aug': aug_name, 'model': model_name, 'mode': 'aug', 'agg': agg_name})
        pd.DataFrame(results).to_csv('./results/' + model_name + '_thresholding_' + agg_name)

def thresholding_acc(model_name, aug_name, agg_name):
    # get aggregated output file using wahtever aggregation you decide on; partial LR probably?
    orig_outputs_file = './outputs/aggregated_outputs/' + model_name + '/orig/mean.h5'
    # TODO: the test postfix is an artifact of forgetting to remove it... doesn't mean anything
    agg_outputs_file = './outputs/aggregated_outputs/' + model_name + '/' + aug_name + '/'+ agg_name + '_test.h5'
    top1s = []
    top5s = []
    orig_top1s = []
    orig_top5s = []
    aug_top1s = []
    aug_top5s = []
    with h5py.File(agg_outputs_file) as hf_agg:
        with h5py.File(orig_outputs_file) as hf_orig:
            for key in tqdm(hf_agg.keys()):
                if 'labels' in key:
                    continue
                agg_outputs = hf_agg[key]
                orig_outputs = hf_orig[key]
                labels = hf_agg[key[:-7] + 'labels']
                
                softmax_agg_outputs = softmax(agg_outputs, axis=1) 
                softmax_orig_outputs = softmax(orig_outputs, axis=1) 
                agg_max = np.max(softmax_agg_outputs,axis=1)
                orig_max = np.max(softmax_orig_outputs,axis=1)
                agg_greater = agg_max > orig_max - .1
                choose_agg = np.where(agg_greater == True)[0]
                choose_orig = np.where(agg_greater == False)[0]
               
                merged_outputs = np.zeros(agg_outputs.shape)
                merged_outputs[choose_agg] = agg_outputs[choose_agg]
                merged_outputs[choose_orig] = orig_outputs[choose_orig]
                # iterate over the two to fill in merged_outputs 
                top1, top5 = evaluate_acc(merged_outputs, labels)
                orig_top1, orig_top5 = evaluate_acc(orig_outputs, labels)
                aug_top1, aug_top5 = evaluate_acc(agg_outputs, labels)
                merged_outputs = torch.Tensor(merged_outputs).cuda()
                labels = torch.Tensor(labels).cuda()
                score = accuracy(merged_outputs, labels, topk=(1,5))
                top1s.append(top1)
                top5s.append(top5)
                orig_top1s.append(orig_top1)
                orig_top5s.append(orig_top5)
                aug_top1s.append(aug_top1)
                aug_top5s.append(aug_top5)
    thresh = (np.mean(top1s), np.mean(top5s))
    orig  = (np.mean(orig_top1s), np.mean(orig_top5s)) 
    aug = (np.mean(aug_top1s), np.mean(aug_top5s))
    return thresh, orig, aug

def evaluate_acc(outputs, labels):
    outputs = torch.Tensor(outputs).cuda()
    labels = torch.Tensor(labels).cuda()
    score = accuracy(outputs, labels, topk=(1,5))
    return score[0].item(), score[1].item()
if __name__ == '__main__':
    restrict_GPU_pytorch(sys.argv[2])
    evaluate_threshold(sys.argv[1]) 
