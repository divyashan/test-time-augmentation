#tta_learn_icml
First, replace 'valdir' and 'traindir' variables in setup.py to a local version of ImageNet.

`python setup.py`

Set up the directory structure for the experiment inputs, outputs, and results.

`python train.py`

Generate the outputs of each step and write the results to dataframes in the ./results directory

`python speed.py`

Produce the speed measurements presented in Section 7.2.

`python plot.py`

This will produce figures included in the submission.

`python class_acc_plots.py`

This will produce the figures in Section 7.3 and the supplement, and save them to the ./figs directory.


