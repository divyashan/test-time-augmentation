import h5py
import numpy as np
from scipy import stats

def propci_wilson_cc(count, nobs, alpha=0.05):
    # get confidence limits for proportion
    # using wilson score method w/ cont correction
    # i.e. Method 4 in Newcombe [1];
    EPS = 1.0
    n = np.maximum(nobs,EPS)
    p = count/n
    q = 1.-p
    z = stats.norm.isf(alpha / 2.)
    z2 = z**2
    denom = 2*(n+z2)
    num = 2.*n*p+z2-1.-z*np.sqrt(z2-2-1./n+4*p*(n*q+1))
    ci_l = num/denom
    num = 2.*n*p+z2+1.+z*np.sqrt(z2+2-1./n+4*p*(n*q-1))
    ci_u = num/denom
    # fix the pathological cases
    if nobs == 0 or count == 0:
        ci_l = 0
    if p == 1 or nobs == 0:
        ci_u = 1
    return ci_l, ci_u

def get_outputs_labels(path):
    hf = h5py.File(path, 'r')
    outputs = []
    labels = []
    for key in hf.keys():
        if 'inputs' in key:
            batch_key = key
            label_key = key[:-7] + "_labels"
            outputs.append(hf[batch_key][:])
            labels.append(hf[label_key][:])
    outputs = np.concatenate(outputs, axis=1)
    labels = np.concatenate(labels)
    return outputs, labels


def get_correct_idxs(preds, labels):
    return np.where(np.argmax(preds, axis=1) == labels)[0]

def get_incorrect_idxs(preds, labels):
    return np.where(np.argmax(preds, axis=1) != labels)[0]

def sort_keys(keys):
    key_ints = [int(k.split('_')[1]) for k in keys]
    return sorted(list(set(key_ints)))
