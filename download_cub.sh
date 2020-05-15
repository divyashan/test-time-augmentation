#!/bin/bash

SAVEDIR='./datasets/cub200'

mkdir -p $SAVEDIR
cd $SAVEDIR

wget "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
tar -xvf CUB_200_2011.tgz && rm CUB_200_2011.tgz

