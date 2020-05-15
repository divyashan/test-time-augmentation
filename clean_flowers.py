#!/usr/bin/env python

import numpy as np
import os
from scipy import io
import pathlib as pl
import pprint
import shutil

FLOWERS_DIR = pl.Path('./datasets/flowers102')
ORIG_IMAGES_DIR = FLOWERS_DIR / 'jpg'
LABELS_PATH = FLOWERS_DIR / 'imagelabels.mat'
SPLITS_PATH = FLOWERS_DIR / 'setid.mat'
OUT_TRAIN_DIR = FLOWERS_DIR / 'train'
OUT_VAL_DIR = FLOWERS_DIR / 'val'
OUT_TEST_DIR = FLOWERS_DIR / 'test'

NUM_IMAGES = 8189
NUM_CLASSES = 102


def main():
    labels_dict = io.loadmat(LABELS_PATH)
    pprint.pprint(labels_dict)
    labels = labels_dict['labels'].ravel()
    assert len(labels) == NUM_IMAGES
    assert len(np.unique(labels)) == NUM_CLASSES
    assert np.min(labels) == 1
    assert np.max(labels) == NUM_CLASSES

    splits_dict = io.loadmat(SPLITS_PATH)
    pprint.pprint(splits_dict)
    train_idxs = splits_dict['trnid'].ravel()
    val_idxs = splits_dict['valid'].ravel()
    test_idxs = splits_dict['tstid'].ravel()

    print("train_idxs shape: ", train_idxs.shape)
    print("val_idxs shape: ", val_idxs.shape)
    print("test_idxs shape: ", test_idxs.shape)

    train_set = set(train_idxs)
    val_set = set(val_idxs)
    test_set = set(test_idxs)

    # mutually exclusive?
    assert len(train_set & val_set) == 0
    assert len(train_set & test_set) == 0
    assert len(val_set & test_set) == 0
    # collectively exhaustive?
    assert len(train_idxs) + len(val_idxs) + len(test_idxs) == NUM_IMAGES
    # one-indexed?
    assert min(train_idxs.min(), val_idxs.min(), test_idxs.min()) == 1

    for i in range(NUM_IMAGES):
        idx = i + 1  # matlab one-indexing
        fname = "image_{:05d}.jpg".format(idx)
        src_path = ORIG_IMAGES_DIR / fname

        if idx in train_set:
            out_dir = OUT_TRAIN_DIR
        elif idx in val_set:
            out_dir = OUT_VAL_DIR
        elif idx in test_set:
            out_dir = OUT_TEST_DIR

        subdir = out_dir / "{:04d}".format(labels[i])
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        dest_path = subdir / fname
        shutil.copy(src_path, dest_path)

    for out_dir in (OUT_TRAIN_DIR, OUT_VAL_DIR, OUT_TEST_DIR):
        # ignore hidden files; eg, .DS_Store on macOS
        subdirs = [f for f in os.listdir(out_dir) if not f.startswith('.')]
        assert len(subdirs) == NUM_CLASSES


if __name__ == '__main__':
    main()
