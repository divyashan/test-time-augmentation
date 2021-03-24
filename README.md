# ICCV 2021 Code Submission

Reproduce the experiments from Paper #9233 with the following instructions. 

### Downloading datasets

### Training models (for Flowers-102)

### Running experiments

Run the following from the root directory: 

```./run_expmts.sh```

This command will run all experiments for the paper. It will create a directory for each dataset's intermediate outputs. Within this directory, there is a folder for the model outputs for each TTA policy. Within each TTA policy folder, the script will write the model outputs for that policy, the aggregated model outputs given a specific aggregation method, and the saved aggregation models. 

Results for each aggregation model are stored under "./results/<dataset-name>/<tta-policy>/<dataset-split>/<aggregation-model-name>".


### Reproducing plots

Figures 1 & 2, which show the number of corruptions and corrections introduced by TTA, can be reproduced via ./notebooks/dataset_section_figures.ipynb.

Figures 3, 4, & 5, which analyze the errors introduced by TTA can be reproduced via ./notebooks/imagenet_errors.ipynb (Figure 3) and ./notebooks/flowers102_errors.ipynb (Figures 4 & 5).

The results tables (and the Latex code for each) can be produced by running ./notebooks/Results Table.ipynb.

Figure 6, which plots the augmentation weights learned by our method on the standard TTA policy, can be reproduced via ./notebooks/analyze_weights.ipynb.

