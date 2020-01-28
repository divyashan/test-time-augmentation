#If pretrained model doesn't exist, load it - in future iterations, we would be training new models
# If outputs on validation data don't exist, produce them
#   To do this, we want to produce all possible permutations of augmentations. This can be done using the ttach package. The augmentations we will consider are the fivecrops, horizontal flips, color jitter, brightness and scaling. This will produce some number of augmentations - not sure 
# If outputs on training data don't exist, produce them
#   Same methodology as above, a million years longer.
# If tta_learn baselines don't exist, produce them + write out test predictions 
# If tta_learn models don't exist, produce them + write out test predictions
# save tta_learn models
import os
import sys
import ssl
import pdb

import ttach as tta
from models import get_pretrained_model
from dataloaders import get_imnet_dataloader
from augmentations import write_augmentation_outputs, write_aug_list, get_single_aug_idxs
from evaluate import write_aggregation_outputs
from ranking import write_ranking_outputs, train_ranked_lrs
from gpu_utils import restrict_GPU_pytorch
from tta_utils import check_if_finished

gpu_arg = sys.argv[2]
restrict_GPU_pytorch(gpu_arg)
ssl._create_default_https_context = ssl._create_unverified_context

train_dir = "/data/ddmg/neuro/datasets/imagenet-first-100-of-each"
val_dir = "/data/ddmg/neuro/datasets/ILSVRC2012/val"
train_output_dir = "./outputs/model_outputs/train"
val_output_dir = "./outputs/model_outputs/val"
ranking_output_dir = "./outputs/ranking_outputs"
aggregated_outputs_dir = "./outputs/aggregated_outputs/"

tta_functions = tta.base.Compose([tta.transforms.FiveCrops(224, 224), tta.transforms.HorizontalFlip(), 
                                  tta.transforms.ColorJitter(), tta.transforms.Rot([2,3])])
#tta_functions = tta.base.Compose([tta.transforms.FiveCrops(224, 224), tta.transforms.HorizontalFlip(), 
#                                  tta.transforms.ColorJitter()])

#model_name = config['model_name']
model_name = sys.argv[1] 
model = get_pretrained_model(model_name)
tta_model = tta.ClassificationTTAWrapperOutput(model, tta_functions, ret_all=True)
tta_model.to('cuda:0')
print("[X] Model loaded!")

# Generate validation outputs
output_file = val_output_dir + "/" + model_name + ".h5"
if not check_if_finished(output_file): 
    # write out testaugmented outputs on val data
    dataloader = get_imnet_dataloader(val_dir, batch_size=4) 
    write_augmentation_outputs(tta_model, dataloader, output_file)
print("[X] Val outputs written!")

# Generate training outputs
output_file = train_output_dir + "/" + model_name + ".h5"
if not check_if_finished(output_file): 
    # write out testaugmented outputs on train data
    dataloader = get_imnet_dataloader(train_dir) 
    write_augmentation_outputs(tta_model, dataloader, output_file)
print("[X] Train outputs written!")

# Learn ranking of set of augmentations
# Methods: OMP, LR
rank_names = ['OMP', 'LR', 'APAC']
aug_names = ['combo']
for rank_name in rank_names:
    for aug_name in aug_names:
        print("RANK ALG: ", rank_name, "\tAUG: ", aug_name)
        if not os.path.exists(ranking_output_dir + "/" + model_name + "/"  + rank_name+ '/'+ aug_name + ".h5"):
            write_ranking_outputs(model_name, aug_name, rank_name)
        if not os.path.exists('./agg_models/ranking/' + aug_name + '/' + rank_name+ '/' + model_name + '/9.pth'):
            train_ranked_lrs(model_name, aug_name, rank_name)
        print("[X] Ranking outputs written!")
        print("[X] Ranked logistic regressions trained!")

# Learn aggregation of augmentation outputs
# Methods: Mean, max, #augmentations-LR, #classes*#augmentations LR
write_aggregation_outputs(model_name)
print("[X] Aggregated outputs written!")

