#!/bin/bash

python train.py resnet18 1
python train.py resnet50 1
python train.py resnet101 1
#python train.py alexnet 1
#python train.py vgg16 1
python train.py MobileNetV2 1
