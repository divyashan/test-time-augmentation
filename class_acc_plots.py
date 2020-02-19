import os
import sys
import h5py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from augmentations import get_aug_idxs
from scipy.special import softmax
from sklearn.metrics import confusion_matrix
from utils.plot_utils import get_binned_changes, get_labels, get_class_accs, plot_class_acc_response
from scipy import stats
sns.set_style('white')
model_name_dict = {'resnet101':'ResNet-101', 'resnet18':'ResNet-18', 'MobileNetV2':'MobileNetV2',
                   'resnet50':'ResNet-50'}
agg_names = ['partial_lr']
aug_names = ['hflip', 'five_crop']
lr_mean_pval = []
lr_orig_pval = []
mean_orig_pval = []
pvalue_dicts = []
all_lr_class_acc = []
all_mean_class_acc = []
all_orig_class_acc = []
for model_name in model_name_dict.keys():
    for agg_name in agg_names:
        for aug_name in aug_names:
            labels = get_labels(model_name, agg_name, aug_name)
            all_true_labels, all_orig_labels, all_mean_labels, all_lr_labels = labels
            orig_class_acc = get_class_accs(all_true_labels, all_orig_labels)[:993]
            mean_class_acc = get_class_accs(all_true_labels, all_mean_labels)[:993]
            lr_class_acc = get_class_accs(all_true_labels, all_lr_labels)[:993]
            lr_mean = stats.ttest_rel(lr_class_acc, mean_class_acc).pvalue 
            lr_orig = stats.ttest_rel(lr_class_acc, orig_class_acc).pvalue
            mean_orig = stats.ttest_rel(orig_class_acc, mean_class_acc).pvalue
            lr_mean_stat = stats.ttest_rel(lr_class_acc, mean_class_acc).statistic
            lr_orig_stat = stats.ttest_rel(lr_class_acc, orig_class_acc).statistic
            mean_orig_stat = stats.ttest_rel(orig_class_acc, mean_class_acc).statistic
            pvalue_dicts.append({'lr_orig_p': lr_orig, 'lr_mean_p':lr_mean, 'mean_orig_p':mean_orig,
                                 'lr_orig_stat': lr_orig_stat, 'lr_mean_stat': lr_mean_stat,
                                 'mean_orig_stat': mean_orig_stat,
                                'model': model_name, 'agg': agg_name, 'aug': aug_name})
            all_mean_class_acc.append(np.mean(mean_class_acc))
            all_lr_class_acc.append(np.mean(lr_class_acc))
            all_orig_class_acc.append(np.mean(orig_class_acc))

from matplotlib.ticker import FormatStrFormatter

sns.set_style('ticks')
model_name_dict = {'resnet101':'ResNet-101', 'resnet18':'ResNet-18', 'MobileNetV2':'MobileNetV2',
                   'resnet50':'ResNet-50'}
model_order = ['MobileNetV2','resnet18', 'resnet50', 'resnet101']
agg_name = 'full_lr'
aug_name = 'hflip'
comp = 'lr'
fig, axs = plt.subplots(ncols=4, squeeze=False, figsize=(15, 3))
axs = axs.flatten()

for i,model_name in enumerate(model_order):
    labels = get_labels(model_name, agg_name, aug_name)
    all_true_labels, all_orig_labels, all_mean_labels, all_lr_labels = labels
    orig_class_acc = np.array(get_class_accs(all_true_labels, all_orig_labels)[:993])
    mean_class_acc = np.array(get_class_accs(all_true_labels, all_mean_labels)[:993])
    lr_class_acc = np.array(get_class_accs(all_true_labels, all_lr_labels)[:993])

    x, y_increase, y_decrease = get_binned_changes(lr_class_acc, orig_class_acc)
    x, mean_y_increase, mean_y_decrease = get_binned_changes(mean_class_acc, orig_class_acc)
    print(np.sum(mean_y_increase), np.sum(y_increase))
    #axs[i].plot(x,y_increase, marker="o", label="Learned TTA (Ours)")
    #axs[i].plot(x,mean_y_increase, marker="o", label="Standard TTA (Baseline)")
    improved_LR = orig_class_acc[np.where(lr_class_acc > orig_class_acc + .05)[0]]
    improved_mean = orig_class_acc[np.where(mean_class_acc > orig_class_acc + .05)[0]]
    print(len(improved_mean), len(improved_LR))
    axs[i].hist(improved_LR, bins=10, range=(0, 1), label="Learned TTA (Ours)")
    axs[i].hist(improved_mean, bins=10, range=(0, 1), label="Standard TTA (Baseline)")
    axs[i].set_ylim(0, 45)
    #axs[i].set_xlim(0, 1)
    axs[i].set_xlabel("Top-1 Acc Before TTA")
    axs[i].set_title(model_name_dict[model_name])

    axs[i].set_xticks(np.arange(0, 1.1, step=.1))
    axs[i].set_xticklabels(np.arange(0, 1.1, step=0.1), rotation=45)
    labels = [item.get_text() for item in axs[i].get_xticklabels()]
    axs[i].set_xticklabels([str(round(float(label), 2)) for label in labels])

    if i == 0:
        axs[i].set_ylabel('Number of Classes')
    else:
        #axs[i].set_ylim(0, 50)
        axs[i].set_yticklabels([])

    if i == 3:
        handles, labels = axs[i].get_legend_handles_labels()
        labels = ['Learned TTA', 'Standard TTA']
        axs[i].legend(handles, labels, loc='upper left', prop={'size': 10})

fig.subplots_adjust(top=0.45)
if comp == 'mean':
    fig.suptitle("Standard Test-Time Augmentation Effect on Original Class Accuracy", size=15, y=1.05)
else:
    fig.suptitle("Frequency with which Test-Time Augmentation Increases Accuracy", size=15, y=1.05)
plt.tight_layout()
plt.savefig('./figs/' + comp + '_class_acc_response_increase.pdf', filetype='pdf', bbox_inches='tight')
plt.clf()


model_order = ['MobileNetV2','resnet18', 'resnet50', 'resnet101']
agg_name = 'partial_lr'
aug_name = 'combo'
comp = 'lr'
fig, axs = plt.subplots(ncols=4, squeeze=False, figsize=(15, 3))
axs = axs.flatten()

for i,model_name in enumerate(model_order):
    labels = get_labels(model_name, agg_name, aug_name)
    all_true_labels, all_orig_labels, all_mean_labels, all_lr_labels = labels
    orig_class_acc = np.array(get_class_accs(all_true_labels, all_orig_labels)[:993])
    mean_class_acc = np.array(get_class_accs(all_true_labels, all_mean_labels)[:993])
    lr_class_acc = np.array(get_class_accs(all_true_labels, all_lr_labels)[:993])

    x, y_increase, y_decrease = get_binned_changes(lr_class_acc, orig_class_acc)
    x, mean_y_increase, mean_y_decrease = get_binned_changes(mean_class_acc, orig_class_acc)
    
    declined_LR = orig_class_acc[np.where(lr_class_acc < orig_class_acc - .05)[0]]
    declined_mean = orig_class_acc[np.where(mean_class_acc < orig_class_acc - .05)[0]]
    axs[i].hist(declined_mean, bins=10, range=(0, 1), label="Standard TTA (Baseline)")
    axs[i].hist(declined_LR, bins=10, range=(0, 1), label="Learned TTA (Ours)")

    axs[i].set_ylim(0, 150)
    #axs[i].set_xlim(0, 1)
    axs[i].set_xlabel("Top-1 Acc Before TTA")
    axs[i].set_title(model_name_dict[model_name])

    axs[i].set_xticks(np.arange(0, 1.1, step=.1))
    axs[i].set_xticklabels(np.arange(0, 1.1, step=0.1), rotation=45)
    labels = [item.get_text() for item in axs[i].get_xticklabels()]
    axs[i].set_xticklabels([str(round(float(label), 2)) for label in labels])

    if i == 0:
        axs[i].set_ylabel('Number of Classes')
    else:
        #axs[i].set_ylim(0, 50)
        axs[i].set_yticklabels([])

    if i == 3:
        handles, labels = axs[i].get_legend_handles_labels()
        labels = ['Standard TTA', 'Learned TTA']
        #labels = ['Learned TTA', 'Standard TTA']
        axs[i].legend(handles, labels, loc='upper left', prop={'size': 10})

fig.subplots_adjust(top=0.45)
fig.suptitle("Frequency with which Test-Time Augmentation Decreases Accuracy (" + aug_name + ")", size=15, y=1.05)
plt.tight_layout()
plt.savefig('./figs/' + comp + '_' + aug_name + '_class_acc_response_decrease.pdf', filetype='pdf', bbox_inches='tight')
plt.clf()
