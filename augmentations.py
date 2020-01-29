import os
import h5py
import torch
import numpy as np
from tqdm import tqdm 
from aug_cartesian_product import AUG_ORDER, AUG_CART_PRODUCT

def write_aug_list(aug_transform_parameters, aug_order):
    def parse_five_crop(param):
        pos_map = {'cr':0, 'lt':1, 'rt':2, 'lb':3, 'rb':4}
        pos  = str(param.func).split('_')[1][:2]
        crop_h = param.keywords['crop_h']
        crop_w = param.keywords['crop_w']
        return pos_map[pos]
    
    def parse_hflip(param):
        return int(param)

    def parse_color_jitter(param):
        return int(param)

    def parse_rotation(param):
        return param.keywords['angle']
    
    parse_fs_dict = {'five_crop': parse_five_crop, 'rotation': parse_rotation,
                'colorjitter': parse_color_jitter, 'hflip': parse_hflip}
    
    parsed_params = []
    parse_fs = [parse_fs_dict[aug] for aug in aug_order]
    for params in aug_transform_parameters:
        parsed = [parse_f(param) for parse_f,param in zip(parse_fs,params)]
        parsed_params.append(parsed)
    return parsed_params

def get_start_ind(output_file, mode, n_batches_per_write):
    with h5py.File(output_file, mode) as hf:
        return len(hf.keys())*n_batches_per_write

def write_augmentation_outputs(tta_model, dataloader, output_file):
    tta_model.eval()
    mode = 'w'
    if os.path.exists(output_file):
        mode = 'r+'
    
    def flush_to_file(i, all_outputs, all_targets):
        if os.path.exists(output_file):
            mode = 'r+'
        with h5py.File(output_file, mode) as hf:
            hf.create_dataset("batch_" + str(i) + "_inputs", compression="gzip", compression_opts=9, 
                              data=all_outputs.astype(np.float16))
            hf.create_dataset("batch_" + str(i) + "_labels", compression="gzip", compression_opts=9,
                              data=all_targets.astype(np.float16))    
        all_outputs.fill(0)
        all_targets.fill(0)
    
    n_augs = len(tta_model.transforms)
    n_examples = len(dataloader.dataset)

    batch_size = dataloader.batch_size
    n_batches_per_write = 100
    n_write= n_batches_per_write*batch_size
    
    all_outputs = np.zeros((n_augs, n_write, 1000))
    all_targets = np.zeros((n_write))

    # Start where TTA write left off
    start_ind = get_start_ind(output_file, mode, n_batches_per_write)
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(dataloader)):
            if i < start_ind:
                continue
            
            if i % n_batches_per_write == 0 and i != 0:
                flush_to_file(i, all_outputs, all_targets)
            
            
            
            images = images.cuda('cuda:0', non_blocking=True)
            output = tta_model(images)
            output = output.cpu().numpy()                 
            target = target.cpu().numpy()
    
            start = (i%n_batches_per_write)*batch_size        
            end = start + min(batch_size,output.shape[1])
            all_outputs[:,start:end,:] = output
            all_targets[start:end] = target

def get_single_aug_idxs(aug_name):
    other_col_idxs = np.where(np.array(AUG_ORDER) != aug_name)[0]        
    other_cols_sum = np.sum(AUG_CART_PRODUCT[:,other_col_idxs], axis=1)
    other_off_idxs = np.where(other_cols_sum == 0)[0]
    return other_off_idxs

def get_original_idxs():
    return np.where(np.sum(AUG_CART_PRODUCT, axis=1) == 0)[0]

def get_all_idxs():
    return np.arange(len(AUG_CART_PRODUCT))

def get_aug_idxs(aug_name):
    if aug_name == 'orig':
        return get_original_idxs()
    elif aug_name == 'combo':
        return get_all_idxs()
    else:
        return get_single_aug_idxs(aug_name)
