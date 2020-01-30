import os
import sys
import ssl
import pdb
import numpy as np
import ttach as tta
from models import get_pretrained_model
from dataloaders import get_imnet_dataloader
from augmentations import write_augmentation_outputs, write_aug_list 
from evaluate import evaluate_aggregation_outputs
from ranking import write_ranking_outputs, train_ranked_lrs, evaluate_ranking
from threshold import evaluate_threshold
from utils.gpu_utils import restrict_GPU_pytorch
from utils.tta_utils import check_if_finished

gpu_arg = sys.argv[2]
restrict_GPU_pytorch(gpu_arg)
ssl._create_default_https_context = ssl._create_unverified_context

train_dir = "/data/ddmg/neuro/datasets/imagenet-first-100-of-each"
val_dir = "/data/ddmg/neuro/datasets/ILSVRC2012/val"
train_output_dir = "./outputs/model_outputs/train"
val_output_dir = "./outputs/model_outputs/val"
ranking_output_dir = "./outputs/ranking_outputs"
ranked_indices_output_dir = "./top_ten_augs"
aggregated_outputs_dir = "./outputs/aggregated_outputs/"

tta_functions = tta.base.Compose([ tta.transforms.FiveCrops(224, 224), tta.transforms.HorizontalFlip(), 
                                  tta.transforms.ColorJitter(), tta.transforms.Rot([2,3])])

# Set up directories
if not os.path.exists(train_output_dir):
    os.makedirs(train_output_dir)
if not os.path.exists(val_output_dir):
    os.makedirs(val_output_dir)
if not os.path.exists(ranking_output_dir):
    os.makedirs(ranking_output_dir)
if not os.path.exists(aggregated_outputs_dir):
    os.makedirs(aggregated_outputs_dir)
if not os.path.exists(ranked_indices_output_dir):
    os.makedirs(ranked_indices_output_dir)

model_name = sys.argv[1] 
model = get_pretrained_model(model_name)
tta_model = tta.ClassificationTTAWrapperOutput(model, tta_functions, ret_all=True)
tta_model.to('cuda:0')

aug_order = ['five_crop', 'hflip', 'colorjitter', 'rotation']
aug_list = write_aug_list(tta_model.transforms.aug_transform_parameters,aug_order)
np.save('./aug_list', aug_list)
np.save('./aug_order', aug_order)
print("[X] Model loaded!")

# Generate validation outputs
# TODO: Move imnet dataloader to be within function for model_name; cleaner train.py file,  
output_file = val_output_dir + "/" + model_name + ".h5"
if not check_if_finished(output_file): 
    dataloader = get_imnet_dataloader(val_dir, batch_size=2) 
    write_augmentation_outputs(tta_model, dataloader, output_file)
print("[X] Val outputs written!")

# Generate training outputs
output_file = train_output_dir + "/" + model_name + ".h5"
if not check_if_finished(output_file): 
    # write out testaugmented outputs on train data
    dataloader = get_imnet_dataloader(train_dir, batch_size=4) 
    write_augmentation_outputs(tta_model, dataloader, output_file)
print("[X] Train outputs written!")

aug_names = ['combo']
rank_names = ['OMP', 'LR', 'APAC']
for rank_name in rank_names:
    for aug_name in aug_names:
        print("RANK ALG: ", rank_name, "\tAUG: ", aug_name)
        if not os.path.exists(ranking_output_dir + "/" + model_name + "/"  + rank_name+ '/'+ aug_name + ".h5"):
            write_ranking_outputs(model_name, aug_name, rank_name)
        if not os.path.exists('./agg_models/ranking/' + aug_name + '/' + rank_name+ '/' + model_name + '/9.pth'):
            train_ranked_lrs(model_name, aug_name, rank_name)
        print("[X] Ranking outputs written!")
        print("[X] Ranked logistic regressions trained!")

# Evaluating each step of the method 
evaluate_ranking(model_name)
print("[X] Ranking outputs evaluated!")

evaluate_aggregation_outputs(model_name)
print("[X] Aggregated outputs written + evaluated!")

evaluate_threshold(model_name)
print("[X] Thresholding evaluated!")
