#!/bin/bash
# setting =

DATASET='flowers102'
N_CLASSES=102
GPU_ARG=3
#python train.py $DATASET $N_CLASSES resnet50 $GPU_ARG 111
#python train.py $DATASET $N_CLASSES resnet18 $GPU_ARG 111
#python train.py $DATASET $N_CLASSES MobileNetV2 $GPU_ARG 111
#python train.py $DATASET $N_CLASSES inceptionv3 $GPU_ARG 111

DATASET='imnet'
N_CLASSES=1000

#python train.py $DATASET $N_CLASSES resnet18 $GPU_ARG 111
#python train.py $DATASET $N_CLASSES resnet50 $GPU_ARG 111
#python train.py $DATASET $N_CLASSES MobileNetV2 $GPU_ARG 111
#python train.py $DATASET $N_CLASSES inceptionv3 $GPU_ARG 011

DATASET='birds200'
N_CLASSES=200
#python train.py $DATASET $N_CLASSES resnet50 $GPU_ARG 011
#python train.py $DATASET $N_CLASSES resnet18 $GPU_ARG 011
#python train.py $DATASET $N_CLASSES MobileNetV2 $GPU_ARG 011
#python train.py $DATASET $N_CLASSES inceptionv3 $GPU_ARG 011

# Training on the additional data and then applying test-time augmentation
DATASET='flowers102'
N_CLASSES=102
GPU_ARG=1
#python train.py $DATASET $N_CLASSES resnet50_added_data $GPU_ARG 111
#python train.py $DATASET $N_CLASSES resnet18_added_data $GPU_ARG 111
#python train.py $DATASET $N_CLASSES MobileNetV2_added_data $GPU_ARG 111
#python train.py $DATASET $N_CLASSES inceptionv3_added_data $GPU_ARG 111

DATASET='birds200'
N_CLASSES=200
#python train.py $DATASET $N_CLASSES resnet50_added_data $GPU_ARG 111
#python train.py $DATASET $N_CLASSES resnet18_added_data $GPU_ARG 111
#python train.py $DATASET $N_CLASSES MobileNetV2_added_data $GPU_ARG 111
#python train.py $DATASET $N_CLASSES inceptionv3_added_data $GPU_ARG 111

DATASET='cifar100'
N_CLASSES=100
python train.py $DATASET $N_CLASSES cifar100_cnn $GPU_ARG 001 

DATASET='stl10'
N_CLASSES=10
GPU_ARG=3
#python train.py $DATASET $N_CLASSES stl10_cnn $GPU_ARG 111
