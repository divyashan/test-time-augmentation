import os
import pdb
import torch
import h5py
import numpy as np
from tqdm import tqdm

from tta_agg_models import TTARegression, TTAPartialRegression, ImprovedLR, TTARegressionFrozen
from utils.imagenet_utils import accuracy, AverageMeter, ProgressMeter
from augmentations import get_aug_idxs
from expmt_vars import val_output_dir, agg_models_dir
from sklearn.metrics import log_loss, roc_auc_score

def train_improved_lr(n_augs, n_classes, train_path, coeffs, n_epochs=10, scale=1):
    ilr = ImprovedLR(n_augs, n_classes, scale)

    hf = h5py.File(train_path, 'r')
    outputs = hf['batch1_inputs'][:]
    labels = hf['batch1_labels'][:]
    n_augs, n_examples, n_classes = outputs.shape

    for i in range(n_classes):
        print("CLASS: ", i)
        class_outputs = outputs[:,:,i]
        class_outputs = np.expand_dims(class_outputs, 2)
        class_labels = np.zeros(labels.shape)
        class_labels[np.where(labels == i)[0]] = 1
        
        # now the input should be a n_augs x n_examples matrix, for class i
        # construct labels as 1 for being that class, 0 if its not
        plr = TTAPartialRegression(n_augs, 1, scale, initialization='even', coeffs=coeffs)
        for j in range(n_epochs):
            loss, auc = train_epoch_BCE(plr, outputs, class_labels, i)
            if j == 0:
                orig_loss, orig_auc = loss, auc
        #weights, _ = nnls(class_outputs.T, class_labels)
        ilr.coeffs[:,i] = plr.coeffs.detach().cpu().numpy()[:,0]
    ilr.coeffs = ilr.coeffs / np.sum(ilr.coeffs, axis=0)
    # TODO: save coeffs 
    return ilr

def train_improved_lr_CE(n_augs, n_classes, train_path, orig_idx, n_epochs=10, scale=1):
    ilr = ImprovedLR(n_augs, n_classes, scale)

    hf = h5py.File(train_path, 'r')
    outputs = hf['batch1_inputs'][:]
    labels = hf['batch1_labels'][:]
    n_augs, n_examples, n_classes = outputs.shape

    for i in range(n_classes):
        print("CLASS: ", i)
        # class_idxs could also subselect for examples predicted to be class i 
        class_idxs = np.where(labels == i)[0]
        pred_idxs = np.where(np.argmax(outputs[orig_idx], axis=1) == i)[0]
        idxs = np.array(list(set(np.concatenate([class_idxs, pred_idxs]))))
        class_outputs = outputs[:,idxs,:]
        class_labels = labels[idxs]
        # now the input should be a n_augs x n_examples matrix, for class i
        # construct labels as 1 for being that class, 0 if its not
        plr = TTAPartialRegression(n_augs, n_classes, scale, initialization='even') 
        for j in range(n_epochs):
            loss, auc = train_epoch_CE(plr, class_outputs, class_labels)
            if j == 0:
                orig_loss, orig_auc = loss, auc
        #weights, _ = nnls(class_outputs.T, class_labels)
        ilr.coeffs[:,i] = plr.coeffs.detach().cpu().numpy()[:,0]
    ilr.coeffs = ilr.coeffs / np.sum(ilr.coeffs, axis=0)
    # TODO: save coeffs 
    return ilr

def train_full_lr_frozen(n_augs, n_classes, train_path,coeffs, n_epochs=20, scale=1):
    hf = h5py.File(train_path, 'r')
    outputs = hf['batch1_inputs'][:]
    labels = hf['batch1_labels'][:]
    n_augs, n_examples, n_classes = outputs.shape

    # class_idxs could also subselect for examples predicted to be class i
    for j in range(n_epochs):
        print("EPOCH: ", j)
        # now the input should be a n_augs x n_examples matrix, for class i
        # construct labels as 1 for being that class, 0 if its not
        flrf = TTARegressionFrozen(n_augs, n_classes, scale, initialization=coeffs)
        flrf.cuda()
        for i in range(n_classes):
            idxs = np.where(labels == i)[0]
            class_outputs = outputs[:,idxs,:]
            class_labels = labels[idxs]
            loss, auc = train_epoch_full_lr_frozen(flrf, class_outputs, class_labels, i)
        #weights, _ = nnls(class_outputs.T, class_labels)
    # TODO: save coeffs
    #model_prefix = agg_models_dir + '/' + model_name + '/' + aug_name
    return flrf

def train_epoch_full_lr_frozen(model, X, y, class_idx):
    n_augs = len(X)
    n_classes = X.shape[2]
    criterion = torch.nn.CrossEntropyLoss()
    # use i to set the number of 
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=.01, momentum=.9, weight_decay=1e-4)
    criterion.cuda('cuda:0')
    model.train()
    params = torch.cat([x.view(-1) for x in model.parameters()])

    X = np.swapaxes(X, 0, 1)
    X = torch.Tensor(X).cuda('cuda:0', non_blocking=True)
    y = torch.Tensor(y).long().cuda('cuda:0', non_blocking=True)
    output = model(X)

    #loss = criterion(class_outputs, y)
    loss = criterion(output, y)
    acc1, _ = accuracy(output, y, topk=(1, 5))
    optimizer.zero_grad()
    loss.backward()
    # Zero out the irrelevant gradients to this class's optimization
    for j in range(n_classes):
        if j != class_idx:
            model.coeffs.grad[:,j] = 0
    optimizer.step()
    for p in model.parameters():
        p.data.clamp_(0)
    return loss.item(), acc1.item()

def train_epoch_CE(model, X, y):
    n_augs = len(X)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=1e-4)
    model.cuda('cuda:0')
    criterion.cuda('cuda:0')
    model.train()
    params = torch.cat([x.view(-1) for x in model.parameters()])

    X = np.swapaxes(X, 0, 1)
    X = torch.Tensor(X).cuda('cuda:0', non_blocking=True)
    y = torch.Tensor(y).long().cuda('cuda:0', non_blocking=True)
    output = model(X)

    #loss = criterion(class_outputs, y)
    loss = criterion(output, y)
    acc1, _ = accuracy(output, y, topk=(1, 5))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    for p in model.parameters():
        p.data.clamp_(0)
    return loss.item(), acc1.item()

def train_epoch_BCE(model, X, y, class_idx):
    n_augs = len(X)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=1e-4)
    model.cuda('cuda:0')
    criterion.cuda('cuda:0')
    model.train()
    params = torch.cat([x.view(-1) for x in model.parameters()])

    X = np.swapaxes(X, 0, 1)
    X = torch.Tensor(X).cuda('cuda:0', non_blocking=True)
    y = torch.Tensor(y).cuda('cuda:0', non_blocking=True)
    
    output = model(X)
    output = torch.softmax(output, axis=1)
    class_outputs = output[:,class_idx]
    
    one_true = np.where(y == 1)[0]
    one_preds = np.where(class_outputs > .5)[0]
    #loss = criterion(class_outputs, y)
    loss = criterion(class_outputs, y)
    acc1, _ = accuracy(output, y, topk=(1, 5))
    np_output = class_outputs.detach().numpy()
    if len(one_preds): 
        acc = len(set(one_preds).intersection(one_true)) / len(one_preds)
    else:
        acc = 0
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    for p in model.parameters():
        p.data.clamp_(0)
    return loss, roc_auc_score(y, np_output) 




def train_tta_lr(model_name, aug_name, epochs, agg_name, dataset, n_classes, temp_scale):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    idxs = get_aug_idxs(aug_name)
    datapath = val_output_dir + '/' + model_name + '_val.h5'
    
    n_augs = len(idxs)
    criterion = torch.nn.CrossEntropyLoss()
    if agg_name == 'full':
        model = TTARegression(n_augs,n_classes,temp_scale,'even')
    elif agg_name == 'partial':
        model = TTAPartialRegression(n_augs,n_classes,temp_scale, 'even')
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=1e-4)

    model.cuda('cuda:0')
    criterion.cuda('cuda:0')
    model.train()
    lambda1 = .01 
    params = torch.cat([x.view(-1) for x in model.parameters()])

    with h5py.File(datapath, 'r') as hf:
        for epoch in range(epochs):
            progress = ProgressMeter(len(hf.keys()),
                            [batch_time, data_time, losses, top1, top5],
                            prefix="Epoch: [{}]".format(epoch))
            examples = hf['batch1_inputs'] 
            target = hf['batch1_labels']
            examples = examples[idxs,:,:]
            examples = np.swapaxes(examples, 0, 1)
            examples = torch.Tensor(examples)
            target = torch.Tensor(target).long()
            if len(examples) < 1000:
                examples = examples.cuda('cuda:0', non_blocking=True)
                target = target.cuda('cuda:0', non_blocking=True)
                output = model(examples)
                nll_loss = criterion(output, target)
                l1_loss = lambda1 * torch.norm(params, 1)
                loss = nll_loss  
                acc1, acc5 = accuracy(output, target, topk=(1,5))

                losses.update(loss.item(), examples.size(0))
                top1.update(acc1[0], examples.size(0))
                top5.update(acc5[0], examples.size(0))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()   
                for p in model.parameters():
                    p.data.clamp_(0)                 
            else:
                n_batches = int(len(examples)/1000 + 1)
                for i in range(n_batches):
                    example_batch = examples[i*1000:(i+1)*1000]
                    target_batch = target[i*1000:(i+1)*1000]
                    if len(target_batch) == 0:
                        continue
                    example_batch = example_batch.cuda('cuda:0', non_blocking=True)
                    target_batch = target_batch.cuda('cuda:0', non_blocking=True)
                    output = model(example_batch)
                    nll_loss = criterion(output, target_batch)
                    l1_loss = lambda1 * torch.norm(params, 1)
                    loss = nll_loss 
                    acc1, acc5 = accuracy(output, target_batch, topk=(1,5))

                    losses.update(loss.item(), examples.size(0))
                    top1.update(acc1[0], examples.size(0))
                    top5.update(acc5[0], examples.size(0))
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()   
                    for p in model.parameters():
                        p.data.clamp_(0)                 
            progress.display(epoch)
        model_prefix = agg_models_dir + '/' + model_name + '/' + aug_name
        if not os.path.exists(model_prefix):
            os.makedirs(model_prefix)
        torch.save(model.state_dict(), model_prefix + '/' + agg_name + '_lr.pth')
    return model

    
