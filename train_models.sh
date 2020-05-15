 #!/bin/bash

DATASET='flowers102'

#python finetune_pretrained.py --dataset $DATASET --model resnet18
#python finetune_pretrained.py --dataset $DATASET --model mobilenet_v2 
python finetune_pretrained.py --dataset $DATASET --model inception_v3

DATASET='birds200'
#python finetune_pretrained.py --dataset $DATASET --model resnet18 
#python finetune_pretrained.py --dataset $DATASET --model mobilenet_v2 
python finetune_pretrained.py --dataset $DATASET --model inception_v3 

