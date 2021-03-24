# ICCV 2021 Code Submission

Reproduce the experiments from Paper #9233 with the following instructions. 

### Downloading datasets

Downloading Flowers-102: 

```
./download_flowers.sh
python clean_flowers.py
```

To download ImageNet, follow the instructions [here][http://www.image-net.org/challenges/LSVRC/2012/index]. We make use of ILSVRC2012 in our experiments. Make sure to set the variables "train_dir" and "val_dir" in setup.py to point to your local ImageNet download.

### Training models (for Flowers-102)

Produce trained models for Flowers-102 by finetuning ImageNet pretrained models by running:

```./train_flowers_models.sh```

### Running experiments

Run all experiments with the following command: 

```./run_expmts.sh```

This will create a directory for each dataset's intermediate outputs. Within this directory, there is a folder for the model outputs for each TTA policy. Within each TTA policy folder, the script will write the model outputs for that policy, the aggregated model outputs given a specific aggregation method, and the saved aggregation models. 

Results for each aggregation model are stored under "./results/<dataset-name>/<tta-policy>/<dataset-split>/<aggregation-model-name>".


### Reproducing plots

Figures 1 & 2, which show the number of corruptions and corrections introduced by TTA, can be reproduced via ./notebooks/dataset_section_figures.ipynb.

Figures 3, 4, & 5, which analyze the errors introduced by TTA can be reproduced via ./notebooks/imagenet_errors.ipynb (Figure 3) and ./notebooks/flowers102_errors.ipynb (Figures 4 & 5).

The results tables (and the Latex code for each) can be produced by running ./notebooks/Results Table.ipynb.

Figure 6, which plots the augmentation weights learned by our method on the standard TTA policy, can be reproduced via ./notebooks/analyze_weights.ipynb.

