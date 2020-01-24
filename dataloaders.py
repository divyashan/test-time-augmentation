import torch
from torchvision import transforms
from torchvision import datasets
import PIL

def get_imnet_dataloader(valdir, batch_size=12, resize_dim=256):
    n_workers = 8
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(resize_dim, interpolation=PIL.Image.BICUBIC),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=n_workers, pin_memory=False)
    
    return val_loader
