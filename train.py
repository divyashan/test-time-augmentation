import os
import sys
import ssl
import pdb
import numpy as np
import ttach as tta
from setup import setup, get_tta_functions_from_aug_order
from utils.gpu_utils import restrict_GPU_pytorch
from utils.tta_utils import check_if_finished, split_val_outputs
from utils.aug_utils import write_aug_list

gen_val_outputs = True 
split_val = True
evaluate = True
gen_train_outputs = False 
threshold = False

dataset = sys.argv[1]
n_classes = int(sys.argv[2])
model_name = sys.argv[3]
gpu_arg = sys.argv[4]
aug_order = ['pil']
#aug_order = ['hflip', 'five_crop', 'scale']
aug_order = ['hflip', 'vflip', 'scale']

parts = sys.argv[5]
gen_val_outputs = True if parts[0] == '1' else False
split_val = True if parts[1] == '1' else False
evaluate = True if parts[2] == '1' else False
gen_train_outputs = False
threshold = False

restrict_GPU_pytorch(gpu_arg)
ssl._create_default_https_context = ssl._create_unverified_context
xx = setup(dataset, n_classes, model_name, aug_order)

import expmt_vars
from expmt_vars  import batch_size, train_dir, val_dir, train_output_dir, val_output_dir
from expmt_vars import aggregated_outputs_dir, aug_order, tta_policy
tta_functions = get_tta_functions_from_aug_order(aug_order, dataset)
model = get_pretrained_model(model_name, dataset)
tta_model = tta.ClassificationTTAWrapperOutput(model, tta_functions, ret_all=True)
tta_model.to('cuda:0')
aug_list = write_aug_list(tta_model.transforms.aug_transform_parameters,aug_order)
np.save('./' + dataset + '/' + tta_policy + '/aug_list.npy', aug_list)
np.save('./' + dataset + '/' + tta_policy + '/aug_order.npy', aug_order)
print("[X] Model loaded!")

from models import get_pretrained_model
from dataloaders import get_dataloader
from augmentations import write_augmentation_outputs
from evaluate import evaluate_aggregation_outputs
from threshold import evaluate_threshold

# Generate validation outputs
if gen_val_outputs:
    output_file = val_output_dir + "/" + model_name + ".h5"
    if not check_if_finished(output_file): 
        dataloader = get_dataloader(dataset, val_dir, batch_size) 
        write_augmentation_outputs(tta_model, dataloader, output_file, n_classes)
    print("[X] Val outputs written!")

# Split validation outputs into 1/2 validation and 1/2 test
if split_val:
    output_file = val_output_dir + "/" + model_name + ".h5"
    split_val_outputs(output_file)
    print("[X} Val outputs split")
# Generate training outputs
if gen_train_outputs:
    output_file =  train_output_dir + "/" + model_name + ".h5"
    if not check_if_finished(output_file): 
        # write out testaugmented outputs on train data
        dataloader = get_dataloader(dataset, train_dir, batch_size) 
        write_augmentation_outputs(tta_model, dataloader, output_file, n_classes)
    print("[X] Train outputs written!")

if evaluate:
    evaluate_aggregation_outputs(model_name, dataset, 'val')
    evaluate_aggregation_outputs(model_name, dataset, 'train')
    print("[X] Aggregated outputs written + evaluated!")

if threshold:
    evaluate_threshold(model_name)
    print("[X] Thresholding evaluated!")
