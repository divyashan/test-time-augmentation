import h5py
import torch
import matplotlib.pyplot as plt
import sys
import csv
from tta_agg_models import TTARegression, TTAPartialRegression
import numpy as np

MODEL_NAMES = {"MobileNetV2": "MobileNetV2", "resnet18": "ResNet-18", "resnet50": "ResNet-50", "resnet101": "ResNet-101"}

def generate_angle_vs_accuracy_data(model_name):
    f = h5py.File('outputs/model_outputs/val/{}.h5'.format(model_name), 'r')

    keys = list(f.keys())
    accuracies = []
    angles = [i for i in range(-15, 16)]
    angles.insert(0, 0)
    
    for i in range(len(angles)):
        acc = 0.
        total = 0.
        for key in keys:
            if "labels" in key:
                continue
            model_outputs = torch.Tensor(f[key][i])
            labels_key = get_labels_key(key)
            labels = torch.Tensor(f[labels_key])
            predicted_labels = torch.argmax(model_outputs, dim=1).float()
            results = torch.sum(labels == predicted_labels)
            acc += float(results)
            total += labels.shape[0]
        
        accuracy = acc/total
        print(accuracy)
        accuracies.append(accuracy)

    return angles[1:], accuracies[1:]

def get_labels_key(key):
    words = key.split("_")
    labels_key = ("_").join(words[:-1]) + "_labels"
    return labels_key


def generate_tta_accuracy_vs_angle_data(model_name):
    f = h5py.File('outputs/model_outputs/val/{}.h5'.format(model_name), 'r')

    keys = list(f.keys())
    accuracies = []
    angles = [i for i in range(-15, 16)]

    for i in range(1, len(angles)-1):
        acc = 0.
        total = 0.
        for key in keys:
            if "labels" in key:
                continue
            smaller_angle_model_outputs = torch.Tensor(f[key][i])
            middle_angle_model_outputs = torch.Tensor(f[key][i+1])
            larger_angle_model_outputs = torch.Tensor(f[key][i+2])

            tta_model_outputs = (smaller_angle_model_outputs + middle_angle_model_outputs + larger_angle_model_outputs)/3
            labels_key = get_labels_key(key)
            labels = torch.Tensor(f[labels_key])
            predicted_labels = torch.argmax(tta_model_outputs, dim=1).float()
            results = torch.sum(labels == predicted_labels)
            acc += float(results)
            total += labels.shape[0]
        
        accuracy = acc/total
        accuracies.append(accuracy)
    
    return angles[1:-1], accuracies

def generate_lr_tta_acc_vs_angle_data(model_name):
    f = h5py.File('outputs/model_outputs/val/{}.h5'.format(model_name), 'r')

    keys = list(f.keys())
    accuracies = []
    angles = [i for i in range(-14, 15)]

    for i in range(len(angles)):
        acc = 0.
        total = 0.
        model_path = '/data/ddmg/neuro/datasets/ILSVRC2012/robustness/robustness/{}/partial_lr/{}.pth'.format(model_name, i)
        model = TTARegression(3, 1000,'even')
        model.load_state_dict(torch.load(model_path))
        model.eval()
        for key in keys:
            if "labels" in key:
                continue
            model_outputs = f[key][i:i+3]
            model_outputs = np.swapaxes(model_outputs, 0, 1)
            model_outputs = torch.Tensor(model_outputs)
            tta_model_outputs = model(model_outputs)
            
            labels_key = get_labels_key(key)
            labels = torch.Tensor(f[labels_key])
            predicted_labels = torch.argmax(tta_model_outputs, dim=1).float()
            results = torch.sum(labels == predicted_labels)
            acc += float(results)
            total += labels.shape[0]
        accuracy = acc/total
        print(accuracy)
        accuracies.append(accuracy)

    return angles, accuracies

def graph_tta_accuracy_vs_angle(all_angles, all_acc, all_tta_angles, all_tta_acc, model_names):
    
    if len(model_names) % 2 == 0:
        num_cols = int(len(model_names)/2)
    else:
        num_cols = int(len(model_names)/2) + 1
    fig, ax = plt.subplots(2, num_cols, sharey=True, sharex=True, figsize=(num_cols*4, 2*4))

    for k in range(len(model_names)):
        i = int(k/num_cols)
        j = k % num_cols
        angles = all_angles[k]
        accuracies = all_acc[k]
        tta_angles = all_tta_angles[k]
        tta_accuracies = all_tta_acc[k]
        model_name = model_names[k]
        ax[i, j].plot(tta_angles, tta_accuracies, label='Mean TTA')
        ax[i, j].plot(angles, accuracies, label='Original Model')
        ax[i, j].set_title(MODEL_NAMES[model_name])
        ax[i, j].legend()
                                                                                                                                            
                                                                                                                                                        
    if len(model_names)% 2 == 1:
        ax[-1, -1].axis('off') 
    fig.text(0.5, 0.001, 'Angle of Applied Rotation', ha='center')
    fig.text(0, 0.5, 'Top1 Accuracy', va='center', rotation='vertical')
    fig.suptitle('Model Performance Across Rotations')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('rot_tta_2.pdf')


if __name__ == "__main__":

    models = ["resnet18"]
    data = []
    for model_name in models:
        angles, accuracies = generate_lr_tta_acc_vs_angle_data(model_name)
        data.append({"Angles": angles, "Model Name": model_name, "Accuracies": accuracies})

    csvfile = "rot_tta_2.csv"
    with open(csvfile, 'w') as csvfile:
        dict_writer = csv.DictWriter(csvfile, data[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(data)

"""
     all_angles = []
     all_acc = []
     all_tta_angles = []
     all_tta_acc = []
     model_names = []
     data = []
     for model_name in MODEL_NAMES:
         angles, accuracies = generate_angle_vs_accuracy_data(model_name)
         all_angles.append(angles)
         all_acc.append(accuracies)
         tta_angles, tta_accuracies = generate_tta_accuracy_vs_angle_data(model_name)
         all_tta_angles.append(tta_angles)
         all_tta_acc.append(tta_accuracies)
         model_names.append(model_name)
         data.append({"Angles": angles, "Accuracies": accuracies, "TTA Angles": tta_angles, "TTA Accuracies": tta_accuracies, "Model Name": model_name})

     csv_name = "rot_tta.csv"

     with open(csv_name, 'w') as csvfile:
         dict_writer = csv.DictWriter(csvfile, data[0].keys())
         dict_writer.writeheader()
         dict_writer.writerows(data)
     graph_tta_accuracy_vs_angle(all_angles, all_acc, all_tta_angles, all_tta_acc, model_names)
"""

     

