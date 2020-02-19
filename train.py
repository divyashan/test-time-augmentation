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
from setup import train_dir, val_dir, train_output_dir, val_output_dir, ranking_output_dir,
                  ranking_output_dir, ranked_indices_output_dir, aggregated_outputs_dir
from utils.gpu_utils import restrict_GPU_pytorch
from utils.tta_utils import check_if_finished

gpu_arg = sys.argv[2]
restrict_GPU_pytorch(gpu_arg)
ssl._create_default_https_context = ssl._create_unverified_context

tta_functions = tta.base.Compose([ tta.transforms.FiveCrops(224, 224), tta.transforms.HorizontalFlip(), 
                                  tta.transforms.ColorJitter(), tta.transforms.Rot([2,3])])

model_name = sys.argv[1] 
model = get_pretrained_model(model_name)
tta_model = tta.ClassificationTTAWrapperOutput(model, tta_functions, ret_all=True)
tta_model.to('cuda:0')
print("[X] Model loaded!")

# Generate validation outputs
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
rank_names = ['LR']
for rank_name in rank_names:
    for aug_name in aug_names:
        print("RANK ALG: ", rank_name, "\tAUG: ", aug_name)
        if not os.path.exists(ranking_output_dir + "/" + model_name + "/"  + rank_name+ '/'+ aug_name + ".h5"):
            write_ranking_outputs(model_name, aug_name, rank_name)
        if not os.path.exists('./agg_models/ranking/' + aug_name + '/' + rank_name+ '/' + model_name + '.pth'):
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
