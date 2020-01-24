from efficientnet_pytorch import EfficientNet
from MobileNetV2 import mobilenet_v2
import sys
sys.path.insert(0, '../')
#from testaug_imagenet.FixRes.imnet_evaluate.resnext_wsl import resnext101_32x48d_wsl, resnext101_32x16d_wsl, resnext101_32x32d_wsl, resnext101_32x48d_wsl

import torch
import torchvision.models as models


MAX_MODEL_BATCH_SIZE = {'resnet': 128, 'alexnet': 128, 'vgg16':128, 'FixResNext101_32x48d_w_train_aug':16, 'FixResNext101_32x48d':16, 'ResNext101_32x48d':16, 'ResNext101_32x16d': 32, 'ResNext101_32x32d': 32, 'EfficientNetB0': 128, 'EfficientNetB1': 64, 'EfficientNetB2': 32, 'EfficientNetB3': 32, 'EfficientNetB4': 32, 'EfficientNetB5': 32, 'EfficientNetB6': 32, 'EfficientNetB7': 16, 'MobileNetV2': 128}

def get_efficientnet_f(net_n):
    def get_model(pretrained=True):
        model = EfficientNet.from_pretrained('efficientnet-b' + str(net_n))
        return model
    return get_model

def get_mobilenet_f():
    def get_model(pretrained=True):
        model = mobilenet_v2(pretrained=True)
        return model
    return get_model

def get_resnext101_32x48d(weight_fname):
    def get_model(pretrained=True):
        model=resnext101_32x48d_wsl(progress=True)
        pretrained_dict=torch.load("pretrained_weights/" + weight_fname,map_location='cpu')['model']

        model_dict = model.state_dict()
        for k in model_dict.keys():
            if(('module.'+k) in pretrained_dict.keys()):
                model_dict[k]=pretrained_dict.get(('module.'+k))
        model.load_state_dict(model_dict)
        return model

    return get_model

def get_pretrained_model(model_name):
    model_f_dict = {}
    model_f_dict = {'resnet18': models.resnet18, 'resnet50': models.resnet50, 'resnet101': models.resnet101,
                    'alexnet': models.alexnet,  'vgg16': models.vgg16} 

    # Adding in efficient_net
    for i in range(8):
        model_f_dict['EfficientNetB' + str(i)] = get_efficientnet_f(i)

    # Adding in mobilenet
    model_f_dict['MobileNetV2'] = get_mobilenet_f()
    return model_f_dict[model_name](pretrained=True)

