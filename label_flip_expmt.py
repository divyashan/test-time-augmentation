import torch
import pandas as pd
from augmentations import get_aug_idxs
from scipy.special import softmax
import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from tta_agg_models import TTARegression, TTAPartialRegression
from utils.gpu_utils import restrict_GPU_pytorch
restrict_GPU_pytorch('0')
from sklearn.metrics import roc_auc_score
import pdb

datasets = ['cifar10', 'stl10', 'cifar100', 'svhn']
n_class_opts = [10, 10, 100, 10]
models = ['cifar10_cnn', 'stl10_cnn', 'cifar100_cnn', 'resnet20']
aug_name = 'combo'
agg_names = ['mean', 'partial_lr', 'full_lr', 'full_lr_svm', 'partial_lr_svm']

"""
datasets = ['stl10']
models = ['stl10_cnn']
n_class_opts = [10]
"""
datasets = ['cifar10', 'stl10', 'svhn']
n_class_opts = [10, 10,  10]
models = ['cifar10_cnn', 'stl10_cnn', 'resnet20']
aug_name = 'combo'
agg_names = ['mean', 'partial_lr', 'full_lr', 'full_lr_svm', 'partial_lr_svm']
def count_changes(test_tta_agg_outputs, orig_outputs, labels):
    correct, corrupt, changed= get_changed_sets(test_tta_agg_outputs, orig_outputs, labels)
    n_correct = len(correct)
    n_corrupt = len(corrupt)
    n_changed = len(changed)
    return n_correct, n_corrupt, n_changed

def get_changed_sets(test_tta_agg_outputs, orig_outputs, labels):
    tta_preds = np.argmax(test_tta_agg_outputs, axis=1)
    orig_preds = np.argmax(orig_outputs, axis=1)
    orig_incorrect = np.where(orig_preds  != labels)[0]
    orig_correct = np.where(orig_preds  == labels)[0]
    tta_correct = np.where(tta_preds == labels)[0]
    tta_incorrect = np.where(tta_preds != labels)[0]
    changed = np.where(tta_preds != orig_preds)[0]
    correct = list(set(tta_correct).intersection(orig_incorrect))
    corrupt = list(set(tta_incorrect).intersection(orig_correct))
    return correct, corrupt, changed

def parse_tta_agg_file(fname):
    test_tta_agg_f = h5py.File(fname)
    tta_output_keys = [x for x in test_tta_agg_f.keys() if 'outputs' in x]
    label_keys = [x[:-7] + 'labels' for x in tta_output_keys]
    
    test_tta_agg_outputs = np.concatenate([test_tta_agg_f[ok][:] for ok in tta_output_keys], axis=0)
    test_tta_agg_outputs = softmax(test_tta_agg_outputs, axis=1)
    labels = np.concatenate([test_tta_agg_f[lk] for lk in label_keys])
    return test_tta_agg_outputs, labels
    
def parse_tta_preagg_file(fname, aug_name):
    test_tta_preagg_f = h5py.File(fname)
    tta_output_keys = [x for x in test_tta_preagg_f.keys() if 'inputs' in x]
    label_keys = [x[:-6] + 'labels' for x in tta_output_keys]
    idxs = get_aug_idxs(aug_name)

    test_tta_outputs = np.concatenate([test_tta_preagg_f[ok][:] for ok in tta_output_keys], axis=1)
    test_tta_outputs = softmax(test_tta_outputs, axis=2)
    test_tta_outputs = test_tta_outputs[idxs]   
    labels = np.concatenate([test_tta_preagg_f[lk] for lk in label_keys])
    return test_tta_outputs, labels

def get_agg_model_weights(model, model_path, n_aug, n_classes):
    if 'partial' in agg_name:
        model = TTAPartialRegression(n_aug+1,n_classes,'even')
    elif 'full' in agg_name:
        model = TTARegression(n_aug+1,n_classes,'even')
    model.load_state_dict(torch.load(model_path))
    coeffs = model.coeffs.detach().cpu().numpy()[1:,:]
    coeffs = np.sum(coeffs, axis=1)
    coeffs = coeffs/np.sum(coeffs)
    return coeffs

def get_clf(preagg_outputs, labels_expanded, n_augs):
    # Takes in a set of preaggregated outputs (AxNxC) and labels
    # where labels describe whether the example was corrupted or corrected
    # TODO: Reshape preagg_outputs to be the right shape for PCA
    # Find the best number of principal components 
    preagg_norm = np.linalg.norm(preagg_outputs, axis=1)
    preagg_outputs = preagg_outputs/ preagg_norm[:,np.newaxis]
    svm = SVC(kernel='poly', degree=2,  C=.001, class_weight='balanced', probability=True)
    svm.fit(preagg_outputs, labels_expanded)

    n_examples = int(len(preagg_outputs)/n_augs)
    pred = svm.predict_proba(preagg_outputs)[:,1] 
    pred = pred.reshape((n_examples, n_augs))
    pred = np.mean(pred, axis=1)
    example_labels = labels_expanded.reshape((n_examples, n_augs))[:,0]
    auc = roc_auc_score(example_labels, pred)
    print("AUC: ", auc)
    differences = []
    threshold_opts = np.arange(0, 1, .05)
    for threshold in threshold_opts:
        keep = np.where(pred > threshold)[0]
        n_ones = len(np.where(example_labels[keep] == 1)[0])
        n_zeros = len(np.where(example_labels[keep] == 0)[0])
        differences.append(n_ones-n_zeros)
    return svm, threshold_opts[np.argmax(differences)], auc

def apply_clf(preagg_outputs, labels_expanded, svm, threshold, n_augs, agg_model_weights):
    #TODO: include changed in preagg outputs and labels so we have a sense
    # of how many images stayed the same
    preagg_norm = np.linalg.norm(preagg_outputs, axis=1)
    preagg_outputs = preagg_outputs/ preagg_norm[:,np.newaxis]

    n_examples= int(len(preagg_outputs)/n_augs)
    pred = svm.predict_proba(preagg_outputs)[:,1] 
    pred = pred.reshape((n_examples, n_augs))

    # TODO: instead of this mean, use the weights to aggregate them
    pred = pred.dot(agg_model_weights) 
    example_labels = labels_expanded.reshape((n_examples, n_augs))[:,0]
    auc = roc_auc_score(example_labels, pred)
    print("AUC: ", auc) 
    keep = np.where(pred > threshold)[0]
    n_ones = len(np.where(example_labels[keep] == 1)[0])
    n_zeros = len(np.where(example_labels[keep] == 0)[0])
    return n_ones, n_zeros, -1, auc

results = []
for i in range(len(datasets)):
    dataset = datasets[i]
    n_classes = n_class_opts[i]
    model = models[i]
    for agg_name in agg_names:
        agg_f_name = agg_name
        if 'svm' in agg_name:
            agg_f_name = agg_name[:-4]
        print("DATASET: ", dataset, "MODEL: ", model, "AGG NAME: ", agg_name)
        idxs = get_aug_idxs(aug_name)
        n_prob_outputs = n_classes*(len(idxs) - 1)
        tta_agg, _ = parse_tta_agg_file("./" + dataset + "/hflip_modified_five_crop_scale/aggregated_outputs/val/" + model + "/" + aug_name + "/" + agg_f_name + ".h5")
        tta_preagg, labels = parse_tta_preagg_file("./" + dataset + "/hflip_modified_five_crop_scale/model_outputs/val/" + model + ".h5", aug_name)
        # TODO: Extend this to when the original index is not 0
        orig = tta_preagg[0]
        tta_preagg = tta_preagg[1:]
        n_examples = orig.shape[0]      
        # separate into validation and test set (random split in half)
        shuffled = np.arange(n_examples) 
        #np.random.shuffle(shuffled)
        pct_val = .75
        train_idxs = shuffled[:int(pct_val*n_examples)]
        test_idxs = shuffled[int(pct_val*n_examples):]
        train_tta_agg, test_tta_agg = tta_agg[train_idxs], tta_agg[test_idxs]
        train_tta_preagg, test_tta_preagg = tta_preagg[:,train_idxs,:], tta_preagg[:,test_idxs,:]
        train_orig, test_orig = orig[train_idxs], orig[test_idxs]
        train_labels, test_labels = labels[train_idxs], labels[test_idxs]
        
        n_correct, n_corrupt, n_changed = count_changes(test_tta_agg, test_orig, test_labels)
        train_auc, test_auc = 0, 0
        if 'svm' in agg_name:
            n_correct, n_corrupt, n_changed = count_changes(train_tta_agg, train_orig, train_labels)
            correct, corrupt, changed = get_changed_sets(train_tta_agg, train_orig, train_labels)
                
            good_train = np.swapaxes(train_tta_preagg[:,correct,:], 0, 1)
            bad_train = np.swapaxes(train_tta_preagg[:,corrupt,:], 0, 1)    
            all_train_preagg = np.concatenate([good_train, bad_train], axis=0)
            n_examples,n_aug,n_probs = all_train_preagg.shape
            print("N training examples: ", n_examples)
            all_train_preagg_reshaped = all_train_preagg.reshape((n_examples*n_aug,n_probs))
            labels_expanded = np.array([1 for i in range(n_correct*n_aug)] + [0 for i in range(n_corrupt*n_aug)])
            
            clf, thresh, train_auc= get_clf(all_train_preagg_reshaped, labels_expanded,n_aug)    
            
            # moving to test set; this should be a function
            model_path = "./" + dataset + "/hflip_modified_five_crop_scale/agg_models/" + model + "/combo/" + agg_name[:-4] + ".pth"
            agg_model_weights = get_agg_model_weights(model, model_path, n_aug, n_classes)
            n_correct, n_corrupt, n_changed = count_changes(test_tta_agg, test_orig, test_labels)
            correct, corrupt, changed = get_changed_sets(test_tta_agg, test_orig, test_labels)
            
            good_test = np.swapaxes(test_tta_preagg[:,correct,:], 0, 1)
            bad_test = np.swapaxes(test_tta_preagg[:,corrupt,:], 0, 1)    
            all_test_preagg = np.concatenate([good_test, bad_test], axis=0)
            n_examples,n_aug,n_probs = all_test_preagg.shape
            print("N testing examples: ", n_examples)
            all_test_preagg_reshaped = all_test_preagg.reshape((n_examples*n_aug,n_probs))
            labels_expanded = np.array([1 for i in range(n_correct*n_aug)] + [0 for i in range(n_corrupt*n_aug)])
            n_correct, n_corrupt, n_changed, test_auc= apply_clf(all_test_preagg_reshaped, labels_expanded, clf, thresh, n_aug,agg_model_weights)
            
        result = {'method': agg_name, 'n_correct': n_correct, 'n_corrupt': n_corrupt,
                  'n_changed': n_changed, 'dataset': dataset, 'model': model, 'aug':aug_name, 'train_auc': train_auc, 'test_auc': test_auc}
        results.append(result)
        pd.DataFrame(results).to_csv('label_flip_results')
