import os
import sys
import numpy as np
import ttach as tta
from models import get_pretrained_model
from augmentations import write_aug_list
from utils.gpu_utils import restrict_GPU_pytorch

train_dir = "/data/ddmg/neuro/datasets/imagenet-first-100-of-each"
val_dir = "/data/ddmg/neuro/datasets/ILSVRC2012/val"
train_output_dir = "./outputs/model_outputs/train100"
val_output_dir = "./outputs/model_outputs/val"
ranking_output_dir = "./outputs/ranking_outputs"
ranked_indices_output_dir = "./top_ten_augs"
aggregated_outputs_dir = "./outputs/aggregated_outputs/"
aug_order = ['five_crop', 'hflip', 'colorjitter', 'rotation']
fig_dir = "./figs"

if __name__ == '__main__':
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
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    model_name = sys.argv[1]
    model = get_pretrained_model(model_name)
    tta_model = tta.ClassificationTTAWrapperOutput(model, tta_functions, ret_all=True)
    tta_model.to('cuda:0')

    aug_list = write_aug_list(tta_model.transforms.aug_transform_parameters,aug_order)
    np.save('./aug_list', aug_list)
    np.save('./aug_order', aug_order)
