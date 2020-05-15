from efficientnet_pytorch import EfficientNet
from MobileNetV2 import mobilenet_v2
from utee import selector
from pytorchcv.model_provider import get_model as ptcv_get_model

import sys
sys.path.insert(0, '../')
#from testaug_imagenet.FixRes.imnet_evaluate.resnext_wsl import resnext101_32x48d_wsl, resnext101_32x16d_wsl, resnext101_32x32d_wsl, resnext101_32x48d_wsl
import pdb
import torch
import torchvision.models as models
from expmt_vars import n_classes
from cnn_finetune import make_model
from mnist_train import Net

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

def get_mnist_model(model_name):
    if 'mnist_cnn' in model_name:
        model = Net()
    elif model_name == 'mnist_1NN':
        input_size = 784
        hidden_size = 500
        num_classes = 10
    model.load_state_dict(torch.load('./saved_models/mnist/'+ model_name + '.pth'))
    return model

def get_flowers_model(model_name):
    m_name = model_name
    if model_name == 'MobileNetV2':
        m_name = 'mobilenet_v2'
    if model_name == 'inceptionv3':
        m_name = 'inception_v3'
    model = make_model(
                    m_name,
                    pretrained=True,
                    num_classes=n_classes,
                    input_size=(224, 224), 
                )
    model.load_state_dict(torch.load('./saved_models/flowers102/' + m_name+ '.pth'))
    return model

def get_birds_model(model_name):
    m_name = model_name
    if model_name == 'MobileNetV2':
        m_name = 'mobilenet_v2'
    if model_name == 'inceptionv3':
        m_name = 'inception_v3'
    model = make_model(
                    m_name,
                    pretrained=True,
                    num_classes=n_classes,
                    input_size=(224, 224), 
                )
    model.load_state_dict(torch.load('./saved_models/birds200/' + m_name+ '.pth'))
    return model
    
def get_svhn_model():
    model = ptcv_get_model("resnet20_svhn", pretrained=True)
    return model

def get_cifar10_model():
    model_raw, _, _= selector.select('cifar10')
    return model_raw

def get_cifar100_model():
    model_raw, _, _= selector.select('cifar100')
    return model_raw

def get_stl10_model():
    model_raw, _, _= selector.select('stl10')
    return model_raw

def get_pretrained_model(model_name, dataset):
    if dataset == 'imnet':
        return get_pretrained_model_imnet(model_name)
    elif dataset == 'cifar10':
        if len(model_name.split('_')) == 3:
            model = get_cifar10_model()
            model.load_state_dict(torch.load('./saved_models/cifar10/' + model_name))
            return model
        else:
            return get_cifar10_model()
    elif dataset == 'cifar100':
        if len(model_name.split('_')) == 3:
            model = get_cifar100_model()
            model.load_state_dict(torch.load('./saved_models/cifar100/' + model_name))
            return model
        else:
            return get_cifar100_model() 
    elif dataset == 'stl10':
        return get_stl10_model()
    elif dataset == 'svhn':
        return get_svhn_model()
    elif dataset == 'flowers102':
        return get_flowers_model(model_name)
    elif dataset == 'birds200':
        return get_birds_model(model_name)
    elif dataset == 'mnist':
        return get_mnist_model(model_name)

def get_pretrained_model_imnet(model_name):
    model_f_dict = {}
    model_f_dict = {'resnet18': models.resnet18, 'resnet50': models.resnet50, 'resnet101': models.resnet101,
                    'alexnet': models.alexnet,  'vgg16': models.vgg16, 'inceptionv3': models.inception_v3, 
                    'densenet161': models.densenet161} 

    # Adding in efficient_net
    for i in range(8):
        model_f_dict['EfficientNetB' + str(i)] = get_efficientnet_f(i)

    # Adding in mobilenet
    model_f_dict['MobileNetV2'] = get_mobilenet_f()
    return model_f_dict[model_name](pretrained=True)

