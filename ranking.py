import os
import pdb
import sys
import h5py
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from tta_train import train_tta_lr
from augmentations import get_aug_idxs
from utils.tta_utils import check_if_finished
from utils.imagenet_utils import accuracy
from tta_agg_models import TTARegression, TTAPartialRegression 

from pyomp.omp import omp
import rankaggregation as ra
from sklearn.linear_model import orthogonal_mp 
from sklearn.preprocessing import OneHotEncoder
from aug_cartesian_product import AUG_CART_PRODUCT

# Take in dataset of augmentations N x A x C (N examples, A augmentations, C classes) and labels (N x C)
# Return ranking for all a_i in A
def write_ranking_outputs(model_name, aug_name, ranking_name):
    outputs_file_train= './outputs/model_outputs/train/' + model_name + '.h5'
    outputs_file_val = './outputs/model_outputs/val/' + model_name + '.h5'
    ranking_outputs_path_train = './ranking_outputs/train/' + model_name + '/' + ranking_name + '/'
    ranking_outputs_path_val = './ranking_outputs/val/' + model_name + '/' + ranking_name + '/'
    ranked_idices = './top_ten_augs/' + model_name + '.txt'
    outputs_files = [outputs_file_train, outputs_file_val]
    ranking_outputs_paths = [ranking_outputs_path_train, ranking_outputs_path_val]

    # Create directories if they don't exist
    for ranking_outputs_path in ranking_outputs_paths:
        if not os.path.exists(ranking_outputs_path):
            os.makedirs(ranking_outputs_path)
    
    # Paths to save ranked outputs (for train & val)
    ranking_outputs_files = [ranking_outputs_paths[0] + aug_name + '.h5',
                             ranking_outputs_paths[1] +  aug_name + '.h5']
    
    # Don't redo work you've already done!
    if np.all([check_if_finished(f) for f in ranking_outputs_files]):
        print("[X] Produced both train & val ranking outputs files")
        return 
    
    # Get ranking
    if ranking_name == 'OMP':
        if os.path.exists(ranked_indices):
            ranking = np.load(ranked_indices)
        else:
            if model_name == 'resnet18':
                ranking = np.array([48, 54, 0, 14, 7, 49, 56, 30, 42, 32])
            elif model_name == 'resnet50':
                ranking = np.array([14, 43, 54, 48, 7, 13, 0, 30, 2, 12])
            elif model_name == 'MobileNetV2':
                ranking = np.array([43, 54, 48, 8, 19, 56, 42, 0, 25, 12])
            ranking = ranking_OMP(model_name, aug_name, 10)
            np.save(ranked_indices, ranking)
        print("OMP RANKING: ", ranking) 
    elif ranking_name == 'LR':
        ranking = ranking_LR(model_name, aug_name, 10)
        print("LR RANKING: ", ranking) 
    
    # Use ranking to produce train & val output files
    aug_idxs = get_aug_idxs(aug_name)
    for outputs_file, ranking_outputs_file in zip(outputs_files, ranking_outputs_files): 
        with h5py.File(outputs_file, 'r') as hf:
            with h5py.File(ranking_outputs_file, 'w') as hf_rank:
                for key in tqdm(hf.keys()):
                    if 'label' in key:
                        continue
                    augmentations = hf[key][aug_idxs,:,:]
                    targets = hf[key[:-7] + '_labels'][:]
                    
                    if ranking_name == 'APAC':
                        ranking = np.random.choice(np.arange(len(aug_idxs)), size=10, replace=False)
                    augmentations = augmentations[ranking]
                    hf_rank.create_dataset(key, data=augmentations)
                    hf_rank.create_dataset(key[:-7] + '_labels', data=targets)

def train_ranked_lrs(model_name, aug_name, ranking_name):
    ranking_outputs_file = './ranking_outputs/train/' + model_name + '/' + ranking_name + '/' + aug_name + '.h5'
    model_save_path = './agg_models/ranking/' + aug_name + '/' + ranking_name + '/' + model_name + '/'

    with h5py.File(ranking_outputs_file, 'r') as hf:
        n_rank_models = hf['batch_100_inputs'].shape[0]
        print("Number of models to learn: ", n_rank_models)
        for i in range(n_rank_models):
            epochs = 5
            criterion = torch.nn.CrossEntropyLoss()
            model = TTAPartialRegression(i+1, 1000, 'even')
            model.cuda()
            criterion.cuda()
            model.train()
            optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=1e-4)
            
             
            for epoch in range(epochs):
                print("EPOCH: ", epoch)
                for key in hf.keys():
                    if 'label' in key:
                        continue
                    examples = np.swapaxes(hf[key][:i+1,:], 0, 1)
                    target = hf[key[:-7] +'_labels']
                    examples = torch.Tensor(examples)
                    target = torch.Tensor(target).long()
                    examples = examples.cuda('cuda:0', non_blocking=True)
                    target = target.cuda('cuda:0', non_blocking=True)

                    output = model(examples)
                    loss = criterion(output, target)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            torch.save(model, model_save_path + str(i) + '.pth')

def evaluate_ranking(model_name):
    aug_names = ['combo']
    rank_names = ['OMP', 'APAC', 'LR']
    results = []
    for aug_name in aug_names:
        for rank_name in rank_names:
            top1s, top5s = evaluate_ranking_aug_rank(model_name, aug_name, rank_name)
            result = [{'top1': top1s[i], 'top5': top5s[i], 'rank': rank_name, 'aug': aug_name,
                       'model': model_name, 'n_augs': i+1} for i in range(10)]
            results.extend(result)
    pd.DataFrame(results).to_csv('./results/' + model_name + '_ranking_fs')

def evaluate_ranking_aug_rank(model_name, aug_name, rank_name):
    ranking_outputs_file = './ranking_outputs/val/' + model_name + '/' + rank_name + '/' + aug_name + '.h5'
    model_save_path = './agg_models/ranking/' + aug_name + '/' + rank_name + '/' + model_name + '/'
    top1s = [[] for i in range(10)]
    top5s = [[] for i in range(10)]
    models = [torch.load(model_save_path + str(i) + '.pth') for i in range(10)]
    with torch.no_grad():
        with h5py.File(ranking_outputs_file, 'r') as hf:
            for key in tqdm(hf.keys()):
                if 'labels' in key:
                    continue
                for i in range(10):
                    model = models[i]
                    outputs = hf[key][:i+1]
                    labels = hf[key[:-7] + '_labels']
                                    
                    outputs = np.swapaxes(outputs, 0, 1)
                    outputs = torch.Tensor(outputs).cuda()
                    agg_outputs = model(outputs)
                    labels = torch.Tensor(labels).cuda()
                    score = accuracy(agg_outputs, labels, topk=(1,5))
                    top1s[i].append(score[0].item())
                    top5s[i].append(score[1].item())
    return [np.mean(rank) for rank in top1s], [np.mean(rank) for rank in top5s]

def ranking_OMP(model_name, aug_name, n_augs):
    outputs_path = './outputs/model_outputs/train/' + model_name + '.h5'
    aug_idxs = get_aug_idxs(aug_name)
    all_rankings = []
    with h5py.File(outputs_path, 'r') as hf:
        for key in tqdm(hf.keys()):
            if 'label' in key:
                continue
            augmentations = hf[key][aug_idxs,:,:]
            augmentations = np.swapaxes(augmentations, 0, 1)
            targets = hf[key[:-7] + '_labels'][:]
            img_ranking = img_ranking_OMP(augmentations, targets, n_augs)
            all_rankings.append(img_ranking)
    rankings = combine_rankings(all_rankings, n_augs)
    return rankings

def img_ranking_OMP(augmentations, targets, n_augs):
    # Given a set of augmentations and target, return 
    # ordered list of augmentations 
    n_targets = len(targets)
    ohe = OneHotEncoder(sparse=False).fit(np.arange(1000).reshape(1,-1).T)
    ohe_targets = ohe.transform(targets.reshape(1, -1).T).reshape((n_targets*1000, -1))
    augmentations = np.swapaxes(augmentations, 1, 2).reshape((n_targets*1000, -1))
    ordered_augs = []
    
    for i in range(n_augs):
        coeffs = omp(augmentations, ohe_targets, ncoef=i+1, verbose=False, maxit=60).coef
        aug_idxs = np.where(coeffs != 0)[0]
        new_aug = list(set(aug_idxs).difference(set(ordered_augs)))
        if len(new_aug) > 0:
            ordered_augs.extend(new_aug)
    return ordered_augs

def ranking_LR(model_name, aug_name, n_augs):
    aug_idxs = get_aug_idxs(aug_name)
    model_path = './agg_models/'+model_name+'/'+aug_name + '/lr.pth'
    if os.path.exists(model_path):
        print("[X] LR Model Trained") 
        model = TTARegression(len(aug_idxs),1000,'even')
        model.load_state_dict(torch.load(model_path))
    else:
        print("[ ] Training LR model")
        model = train_tta_lr(model_name, aug_name, 5) 
    coeffs = model.coeffs.detach().numpy()
    augmentation_coeffs = np.sum(coeffs, axis=1)
    rankings = np.flip(np.argsort(augmentation_coeffs))
    return rankings[:n_augs]
 
def combine_rankings(rankings, n_augs):
    # rankings is a N x A matrix of where row N[i] describes the
    # ranking of teh A augmentations for example X_i
    agg = ra.RankAggregator()
    ranked_augs = agg.average_rank(rankings)
    ranks = [x[0] for x in sorted(ranked_augs, key=lambda x: x[1])]
    return ranks[:n_augs]

if __name__=='__main__':
    evaluate_ranking(sys.argv[1])
