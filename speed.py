import pdb
import sys
import time
import h5py 
import torch
import numpy as np
import ttach as tta
import pandas as pd

from tqdm import tqdm
from models import get_pretrained_model
from dataloaders import get_imnet_dataloader

from tta_train import train_tta_lr
from utils.gpu_utils import restrict_GPU_pytorch
from expmt_vars import val_dir
restrict_GPU_pytorch(sys.argv[1])

dataloader = get_imnet_dataloader(val_dir, batch_size=1)
model_names = ['resnet18', 'resnet50', 'resnet101', 'MobileNetV2']
agg_names =  ['partial_lr']
aug_names = ['hflip', 'five_crop', 'combo']

tta_functions_map = {'hflip':tta.base.Compose([tta.transforms.HorizontalFlip()]),
                     'five_crop': tta.base.Compose([tta.transforms.FiveCrops(224, 224)]),
                     'combo':tta.base.Compose([tta.transforms.FiveCrops(224, 224), tta.transforms.HorizontalFlip(),tta.transforms.ColorJitter(), tta.transforms.Rot([2,3])])}
for model_name in model_names:
    results = []
    for aug_name in aug_names:
        print('MODEL: ', model_name, '\tAUG: ', aug_name)
        print("Training partial lr...")
        start = time.time()
        partial_lr_model = train_tta_lr(model_name, aug_name, 5, 'partial')
        end = time.time()
        partial_lr_train_time = end-start
        
        print("Training full lr...")
        start = time.time()
        full_lr_model = train_tta_lr(model_name, aug_name, 5, 'full')
        end = time.time()
        full_lr_train_time = end-start
        
        # Time speed of original inference
        # Time speed of batch inference
        model = get_pretrained_model(model_name)
        tta_functions = tta_functions_map[aug_name]
        tta_model = tta.ClassificationTTAWrapperOutput(model, tta_functions, ret_all=True)
        model.cuda()
        tta_model.cuda()
        partial_lr_model.eval()
        full_lr_model.eval()
        print("Timing inference...")
        with torch.no_grad():
            orig_inf_times = []
            partial_lr_inf_times = []
            full_lr_inf_times = []
            for i, (images, target) in enumerate(tqdm(dataloader)):
                images = images.cuda('cuda:0', non_blocking=True)
                
                def get_full_lr_inf_time():
                    start = time.time()
                    new_outputs = tta_model(images)
                    new_outputs = np.swapaxes(new_outputs.cpu().numpy(), 0, 1)
                    new_outputs = torch.Tensor(new_outputs).cuda()
                    agg_outputs = full_lr_model(new_outputs)
                    end = time.time()
                    return end-start
               
                def get_orig_inf_time(): 
                    start = time.time()
                    orig_outputs = model(images)
                    end = time.time()
                    return end-start 
                
                def get_partial_lr_inf_time():
                    start = time.time()
                    new_outputs = tta_model(images)
                    new_outputs = np.swapaxes(new_outputs.cpu().numpy(), 0, 1)
                    new_outputs = torch.Tensor(new_outputs).cuda()
                    agg_outputs = partial_lr_model(new_outputs)
                    end = time.time()
                    return end-start
                
                order = ['full_lr', 'partial_lr', 'orig']
                fs = [get_full_lr_inf_time, get_partial_lr_inf_time, get_orig_inf_time]
                idxs = np.random.permutation(3)
                new_order = [order[x] for x in idxs]
                new_fs = [fs[x] for x in idxs]
                result_dict = {}
                for name, f in zip(new_order, new_fs):
                     result_dict[name] = f()

                orig_inf_times.append(result_dict['orig'])
                partial_lr_inf_times.append(result_dict['partial_lr'])
                full_lr_inf_times.append(result_dict['full_lr'])
                # Only need one iteration
                # TODO: might be better to average over 10
                if i == 100:
                    break
            orig_inf_mean, orig_inf_std  = np.mean(orig_inf_times), np.std(orig_inf_times)
            partial_lr_inf_mean, partial_lr_std= np.mean(partial_lr_inf_times), np.std(partial_lr_inf_times)
            full_lr_inf_mean, full_lr_std = np.mean(full_lr_inf_times), np.std(full_lr_inf_times)
            
            partial_lr_diffs = np.array(partial_lr_inf_times) - np.array(orig_inf_times)
            full_lr_diffs = np.array(full_lr_inf_times) - np.array(orig_inf_times)

            partial_lr_diff_mean, partial_lr_diff_std  = np.mean(partial_lr_diffs), np.std(partial_lr_diffs)
            full_lr_diff_mean, full_lr_diff_std = np.mean(full_lr_diffs), np.std(full_lr_diffs)

            partial_result = {'aug':aug_name, 'model':model_name, 'agg': 'partial_lr', 
                              'orig_inf':orig_inf_mean,'orig_inf_std':orig_inf_std,
                              'inf_mean':partial_lr_inf_mean, 'inf_std':partial_lr_std,
                              'inf_diff_mean': partial_lr_diff_mean, 'inf_diff_std':partial_lr_diff_std,
                              'train': partial_lr_train_time}
            
            full_result = {'aug':aug_name, 'model':model_name, 'orig_inf':orig_inf_mean, 'orig_inf_std':orig_inf_std,
                        'agg': 'full_lr', 'new_inf':full_lr_inf_mean, 'new_inf_std':full_lr_std,
                         'inf_diff': full_lr_inf_mean-orig_inf_mean, 'train': full_lr_train_time}
            print(partial_result)
            print(full_result)
            results.extend([partial_result, full_result])
            pd.DataFrame(results).to_csv('./results/' + model_name + '_speed_df')         


