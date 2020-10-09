import pdb
import os
import time
from utee import misc
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np

from utils.gpu_utils import restrict_GPU_pytorch
epochs = 50
lr = 0.001
test_interval = 5
decreasing_lr = [80, 120]
log_interval = 41
def train_f(model, train_loader):
    model.cuda()
    losses = []
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_loss = []
        if epoch in decreasing_lr:
            optimizer.param_groups[0]['lr'] *= 0.1
        for batch_idx, (data, target) in enumerate(train_loader):
            indx_target = target.clone()
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
            if batch_idx % log_interval == 0 and batch_idx > 0:
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct = pred.cpu().eq(indx_target).sum()
                acc = correct * 1.0 / len(data)
                print('Train Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.4f} lr: {:.2e}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    loss.item(), acc, optimizer.param_groups[0]['lr'])) 
        losses.append(np.mean(epoch_loss))
    return losses
