 #!/bin/bash

DATASET='flowers102'

python finetune_pretrained.py --dataset $DATASET --model resnet18 --add True --gpu $GPU_ARG
python finetune_pretrained.py --dataset $DATASET --model resnet50 --add True --gpu $GPU_ARG
python finetune_pretrained.py --dataset $DATASET --model mobilenet_v2 --add True --gpu $GPU_ARG
python finetune_pretrained.py --dataset $DATASET --model inception_v3 --add True --gpu $GPU_ARG

DATASET='birds200'
python finetune_pretrained.py --dataset $DATASET --model resnet18 --add True --gpu $GPU_ARG
python finetune_pretrained.py --dataset $DATASET --model resnet50 --add True --gpu $GPU_ARG
python finetune_pretrained.py --dataset $DATASET --model mobilenet_v2 --add True --gpu $GPU_ARG
python finetune_pretrained.py --dataset $DATASET --model inception_v3 --add True --gpu $GPU_ARG

#DATASET='flowers102'
#python finetune_pretrained.py --dataset $DATASET --model resnet18
#python finetune_pretrained.py --dataset $DATASET --model resnet50
#python finetune_pretrained.py --dataset $DATASET --model mobilenet_v2 
#python finetune_pretrained.py --dataset $DATASET --model inception_v3

#DATASET='birds200'
#python finetune_pretrained.py --dataset $DATASET --model resnet18 
#python finetune_pretrained.py --dataset $DATASET --model resnet50 
#python finetune_pretrained.py --dataset $DATASET --model mobilenet_v2 
#python finetune_pretrained.py --dataset $DATASET --model inception_v3 

