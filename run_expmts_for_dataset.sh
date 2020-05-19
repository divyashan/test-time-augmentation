 #!/bin/bash
# setting =

DATASET='flowers102'
N_CLASSES=102
GPU_ARG=1
#python train.py $DATASET $N_CLASSES resnet50 $GPU_ARG 011
#python train.py $DATASET $N_CLASSES resnet18 $GPU_ARG 011
#python train.py $DATASET $N_CLASSES MobileNetV2 $GPU_ARG 011
#python train.py $DATASET $N_CLASSES inceptionv3 $GPU_ARG 011
 
DATASET='imnet'
N_CLASSES=1000

#python train.py $DATASET $N_CLASSES resnet18 $GPU_ARG 011
#python train.py $DATASET $N_CLASSES resnet50 $GPU_ARG 011
#python train.py $DATASET $N_CLASSES MobileNetV2 $GPU_ARG 011
#python train.py $DATASET $N_CLASSES inceptionv3 $GPU_ARG 011

DATASET='birds200'
N_CLASSES=200
#python train.py $DATASET $N_CLASSES resnet50 $GPU_ARG 011
#python train.py $DATASET $N_CLASSES resnet18 $GPU_ARG 011
#python train.py $DATASET $N_CLASSES MobileNetV2 $GPU_ARG 011
python train.py $DATASET $N_CLASSES inceptionv3 $GPU_ARG 111
