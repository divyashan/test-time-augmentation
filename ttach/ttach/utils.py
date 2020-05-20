import torch
from torchvision import transforms
import pdb

means_dict = {'imnet': (0.485, 0.456, 0.406),
             'flowers102': (0.5208, 0.4205, 0.3441),
             'birds200': (0.485, 0.456, 0.406),
             'mnist': (1, 1, 1)}
stds_dict = {'imnet': (0.229, 0.224, 0.225),
             'flowers102': (0.2944, 0.2465, 0.2735),
             'birds200': (0.229, 0.224, 0.225),
             'mnist': (1, 1, 1)}

def pil_wrap(img, stds, means):
  """Convert the `img` numpy tensor to a PIL Image."""
  img_u = img.cpu() * torch.tensor(stds).view(3, 1, 1)
  img_u = img_u + torch.tensor(means).view(3, 1, 1)
  img = transforms.ToPILImage()(img_u).convert("RGB")
  return img

def pil_unwrap(pil_img, stds, means):
  """Converts the PIL img to a numpy array."""
  img_u = pil_img.convert("RGB")
  img_u = transforms.ToTensor()(img_u)
  try:
      img_u = img_u - torch.tensor(means).view(3, 1, 1)
      img_n = img_u / torch.tensor(stds).view(3, 1, 1)
  except:
      pdb.set_trace()
  return img_n.unsqueeze(0).cuda()

def pil_wrap_imgs(imgs, dataset):
  stds = stds_dict[dataset]
  means = means_dict[dataset]
  pil_imgs = [pil_wrap(img, stds, means) for img in imgs]
  return pil_imgs

def pil_unwrap_imgs(imgs, dataset):
  stds = stds_dict[dataset]
  means = means_dict[dataset]
  tensor_imgs = [pil_unwrap(img, stds, means) for img in imgs]
  return torch.cat(tensor_imgs)
