import os
import pdb
import torch
import h5py
import numpy as np
from tqdm import tqdm

from tta_agg_models import TTARegression, TTAPartialRegression
from utils.imagenet_utils import accuracy, AverageMeter, ProgressMeter
from augmentations import get_aug_idxs
from expmt_vars import val_output_dir, agg_models_dir

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
                loss = criterion(output, target)
                acc1, acc5 = accuracy(output, target, topk=(1,5))

                losses.update(loss.item(), examples.size(0))
                top1.update(acc1[0], examples.size(0))
                top5.update(acc5[0], examples.size(0))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()   
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
                    loss = criterion(output, target_batch)
                    acc1, acc5 = accuracy(output, target_batch, topk=(1,5))

                    losses.update(loss.item(), examples.size(0))
                    top1.update(acc1[0], examples.size(0))
                    top5.update(acc5[0], examples.size(0))
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()   
                    
            progress.display(epoch)
        model_prefix = agg_models_dir + '/' + model_name + '/' + aug_name
        if not os.path.exists(model_prefix):
            os.makedirs(model_prefix)
        torch.save(model.state_dict(), model_prefix + '/' + agg_name + '_lr.pth')
    return model

    
