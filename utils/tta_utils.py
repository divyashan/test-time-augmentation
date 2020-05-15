import numpy as np
import h5py
import os
import pdb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import torch
from torch import nn, optim
from scipy.special import softmax

means_dict = {'imnet': (0.485, 0.456, 0.406),
             'flowers102': (0.5208, 0.4205, 0.3441),
             'birds200': (0.485, 0.456, 0.406)}
stds_dict = {'imnet': (0.229, 0.224, 0.225),
             'flowers102': (0.2944, 0.2465, 0.2735),
             'birds200': (0.229, 0.224, 0.225)}
# Check if file is done being written
def check_if_finished(file_path):
    done = False
    if os.path.exists(file_path):
        with h5py.File(file_path, 'r') as hf:
            if 'val' in file_path:
                done = len(hf.keys()) == 248
                if 'Efficient' in file_path:
                    done = len(hf.keys()) == 294
            if 'train' in file_path:
                done = len(hf.keys()) == 82
                if 'Efficient' in file_path:
                    done = len(hf.keys()) == 498
    return done

def split_val_outputs(file_path):
    with h5py.File(file_path) as hf:
        output_keys = [x for x in hf.keys() if 'inputs' in x]
        label_keys = [x[:-6] + 'labels' for x in output_keys]
        pdb.set_trace()
        outputs = np.concatenate([hf[ok][:] for ok in output_keys], axis=1)
        labels = np.concatenate([hf[lk][:] for lk in label_keys])
        outputs = np.swapaxes(outputs, 0, 1)

        # balanced split using sklearn? then write to two separate files?
        X_train, X_test, y_train, y_test = train_test_split(outputs, labels, test_size=.5, random_state=42)
        X_train = np.swapaxes(X_train, 0, 1)
        X_test = np.swapaxes(X_test, 0, 1)
        
        val_file_path = file_path[:-3] + '_val.h5'
        val_hf = h5py.File(val_file_path, 'w')
        val_hf['batch1_inputs'] = X_train
        val_hf['batch1_labels'] = y_train
        val_hf.close()
        
        test_file_path = file_path[:-3] + '_test.h5'
        test_hf = h5py.File(test_file_path, 'w')
        test_hf['batch1_inputs'] = X_test
        test_hf['batch1_labels'] = y_test
        test_hf.close()
        return 

def get_calibration(train_path, orig_idx):
    with h5py.File(train_path) as hf:
        outputs = hf['batch1_inputs'][:]
        labels = hf['batch1_labels'][:]
        outputs = outputs[orig_idx][0]
        outputs = torch.Tensor(outputs).cuda()
        labels = torch.Tensor(labels).long().cuda()
        
        ts_model = TemperatureScaling()
        ts_model = ts_model.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        before_temperature_nll = nll_criterion(outputs, labels).item()
        optimizer = optim.LBFGS([ts_model.temperature], lr=0.01, max_iter=50)
        def eval():
            loss = nll_criterion(ts_model(outputs), labels)
            loss.backward()
            return loss
        optimizer.step(eval)
        after_temperature_nll = nll_criterion(ts_model(outputs), labels).item()
        print(before_temperature_nll, after_temperature_nll, ts_model.temperature)
        temp = ts_model.temperature.detach().cpu().numpy()[0]
        return temp

class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        temperature = self.temperature.unsqueeze(1).expand(x.size(0), x.size(1))
        return x/temperature

def pil_wrap(img, stds, means):
  """Convert the `img` numpy tensor to a PIL Image."""
  img_u = img * torch.tensor(std).view(3, 1, 1)
  img_u = img_u + torch.tensor(mean).view(3, 1, 1)
  img = transforms.ToPILImage()(img_u)
  return img

def pil_unwrap(pil_img, stds, means):
  """Converts the PIL img to a numpy array."""
  img_u = transforms.ToTesnor()(pil_img)
  img_u = img_u - torch.tensor(mean).view(3, 1, 1)
  img_n = img_u / torch.tensor(std).view(3, 1, 1)
  return img_n

def pil_wrap_imgs(imgs, dataset):
  stds = std_dict[dataset]
  means = mean_dict[dataset]
  # 
  return

def pil_unwrap_imgs(imgs, dataset):
  stds = std_dict[dataset]
  means = mean_dict[dataset]
  #
  return
