#!/usr/bin/env python

import os
import pandas as pd
import pathlib as pl
import shutil

DATASET_DIR = pl.Path('./datasets/cub200')
ORIG_IMAGES_DIR = DATASET_DIR / 'CUB_200_2011' / 'images'
IMAGES_INDEX_PATH = DATASET_DIR / 'CUB_200_2011' / 'images.txt'
SPLITS_PATH = DATASET_DIR / 'CUB_200_2011' / 'train_test_split.txt'
OUT_TRAIN_DIR = DATASET_DIR / 'train'
OUT_TEST_DIR = DATASET_DIR / 'test'

NUM_IMAGES = 11788
NUM_CLASSES = 200


def main():
    is_train = pd.read_csv(SPLITS_PATH, sep=' ', index_col=0, header=None,
                           names=['is_train'])
    assert is_train.shape == (NUM_IMAGES, 1)

    rel_paths = pd.read_csv(IMAGES_INDEX_PATH, sep=' ', index_col=0,
                            header=None, names=['rel_path'])
    assert rel_paths.shape == (NUM_IMAGES, 1)

    # why the heck would you make me join on the image number? why would you
    # possibly not just include this all in one file? and why would you not
    # just split it into train and test to begin with?
    df = rel_paths.join(is_train)

    for rel_path, is_train in zip(df['rel_path'], df['is_train']):
        subdir = rel_path.split('/')[0]
        out_dir = OUT_TRAIN_DIR if is_train else OUT_TEST_DIR
        out_subdir = out_dir / subdir
        if not os.path.exists(out_subdir):
            os.makedirs(out_subdir)

        src_path = ORIG_IMAGES_DIR / rel_path
        dest_path = out_dir / rel_path
        shutil.copy(src_path, dest_path)

    for out_dir in (OUT_TRAIN_DIR, OUT_TEST_DIR):
        # ignore hidden files; eg, .DS_Store on macOS
        subdirs = [f for f in os.listdir(out_dir) if not f.startswith('.')]
        assert len(subdirs) == NUM_CLASSES


if __name__ == '__main__':
    main()
