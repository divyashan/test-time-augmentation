# Given a model_name + aug_name, apply thresholding to the aggregated outputs
# Where thresholding is using the original prediction depending on the predicted probability
from imagenet_utils import accuracy
from aug_cartesian_product import AUG_ORDER
import pandas as pd

def evaluate_thresholding(model_name):
    aug_names = AUG_ORDER
    results = []
    for aug_name in aug_names:
        top1, top5 = thresholding_acc(model_name, aug_name) 
        results.append({'top1': top1, 'top5': top5, 'aug': aug_name, 'model': model})
    pd.DataFrame(results).to_csv('./results/' + model_name + '_thresholding')

def thresholding_acc(model_name, aug_name):
    # get aggregated output file using wahtever aggregation you decide on; partial LR probably?
    orig_outputs_file = 
    agg_outputs_file = 

    top1s = []
    top5s = []
    with h5py.File(agg_utputs_file) as hf_agg:
        with h5py.File(orig_utputs_file) as hf_orig:
            for key in hf.keys():
                if 'labels' in key:
                    continue
                agg_outputs = hf_agg[key]
                orig_outputs = hf_orig[key]
                labels = hf_agg[key[:-7] + '_labels']
                
                merged_outputs = np.zeros(agg_outputs.shape)
                pdb.set_trace()
                # iterate over the two to fill in merged_outputs 
                score = accuracy(merged_outputs, labels)
                top1s.append(score[0].item())
                top5s.append(score[1].item())
    return np.mean(top1s), np.mean(top5s)
   
