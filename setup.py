import os
import sys
import numpy as np
import ttach as tta
from models import get_pretrained_model
sys.path.insert(0, './')

from utils.aug_utils import write_aug_list
from utils.gpu_utils import restrict_GPU_pytorch
import pdb
import pickle

def get_tta_functions_from_aug_order(aug_order, dataset):
    crop_size = 224
    if aug_order[0] == 'pil':
        tta_functions = tta.base.Compose([ tta.transforms.AllPIL(crop_size, dataset)])
        return tta_functions
    transform_map = {'hflip': tta.transforms.HorizontalFlip(),
                             'five_crop': tta.transforms.FiveCrops(crop_size, crop_size),
                             'scale': tta.transforms.Scale([1.04, 1.10]),
                             'modified_five_crop': tta.transforms.ModifiedFiveCrops(crop_size, crop_size)}
    fns = [transform_map[x] for x in aug_order]
    tta_functions = tta.base.Compose(fns)
    return tta_functions

def setup(dataset, n_classes, model_name, aug_order):
    crop_size = 224
    batch_size = 2 

    train_dir = "/data/ddmg/neuro/datasets/" + dataset + "/train/"
    val_dir = "/data/ddmg/neuro/datasets/" + dataset + "/val"
    if dataset == 'imnet':
        train_dir = "/data/ddmg/neuro/datasets/ILSVRC2012/train"
        val_dir = "/data/ddmg/neuro/datasets/ILSVRC2012/val"
    #aug_order = ['pil']
    #aug_order = ['hflip', 'modified_five_crop', 'scale']
    tta_policy = '_'.join(sorted(aug_order))
    train_output_dir = "./" + dataset + "/" + tta_policy + "/model_outputs/train"
    val_output_dir = "./" + dataset + "/" + tta_policy + "/model_outputs/val"
    #ranking_output_dir = "./ranking_outputs"
    #ranked_indices_output_dir = "./top_ten_augs"
    aggregated_outputs_dir = "./" + dataset + "/" + tta_policy + "/aggregated_outputs/"
    agg_models_dir = "./" + dataset + "/" + tta_policy + "/agg_models"
    fig_dir = "./figs"
    results_dir = './results/' + dataset + '/' + tta_policy 

    expmt_vars_dict = {'dataset': dataset,
                       'n_classes': n_classes,
                       'model_name': model_name,
                       'batch_size': batch_size,
                       'aug_order_concat': ','.join(aug_order),
                       'train_dir': train_dir,
                       'val_dir': val_dir,
                       'train_output_dir': train_output_dir,
                       'val_output_dir': val_output_dir,
                       'aggregated_outputs_dir': aggregated_outputs_dir,
                       'results_dir': results_dir, 
                       'tta_policy': tta_policy,
                       'agg_models_dir': agg_models_dir}

    with open('vars.pickle', 'wb') as handle:
        pickle.dump(expmt_vars_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    tta_functions = get_tta_functions_from_aug_order(aug_order, dataset)

    # Set up directories

    if not os.path.exists(train_output_dir):
        os.makedirs(train_output_dir)
    if not os.path.exists(val_output_dir):
        os.makedirs(val_output_dir)
    if not os.path.exists(aggregated_outputs_dir):
        os.makedirs(aggregated_outputs_dir)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    if not os.path.exists(agg_models_dir):
        os.makedirs(agg_models_dir)
    if not os.path.exists(results_dir + '/val'):
        os.makedirs(results_dir + '/val')
        os.makedirs(results_dir + '/train')
    model = get_pretrained_model(model_name, dataset)
    tta_model = tta.ClassificationTTAWrapperOutput(model, tta_functions, ret_all=True)
    tta_model.to('cuda:0')
    aug_list = write_aug_list(tta_model.transforms.aug_transform_parameters,aug_order)
    np.save('./' + dataset + '/' + tta_policy + '/aug_list.npy', aug_list)
    np.save('./' + dataset + '/' + tta_policy + '/aug_order.npy', aug_order)
