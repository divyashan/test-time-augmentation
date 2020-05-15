import torch
import torch.nn as nn
from typing import Optional, Mapping, Union
import pdb
from .base import Merger, Compose
import numpy as np


class ClassificationTTAWrapper(nn.Module):
    """Wrap PyTorch nn.Module (classification model) with test time augmentation transforms

    Args:
        model (torch.nn.Module): classification model with single input and single output
            (.forward(x) should return either torch.Tensor or Mapping[str, torch.Tensor])
        transforms (ttach.Compose): composition of test time transforms
        merge_mode (str): method to merge augmented predictions mean/gmean/max/min/sum/tsharpen
        output_mask_key (str): if model output is `dict`, specify which key belong to `label`
    """

    def __init__(
        self,
        model: nn.Module,
        transforms: Compose,
        merge_mode: str = "mean",
        output_label_key: Optional[str] = None,
        ret_all: bool = False 
    ):
        super().__init__()
        print("INITIALIZED")
        self.model = model
        self.transforms = transforms
        self.merge_mode = merge_mode
        self.output_key = output_label_key
        self.ret_all = ret_all

    def forward(
        self, image: torch.Tensor, *args
    ) -> Union[torch.Tensor, Mapping[str, torch.Tensor]]:
        merger = Merger(type=self.merge_mode, n=len(self.transforms))

        print("Performing forward")
        augmented_images = []
        n_transforms = len(self.transforms)
        for transformer in self.transforms:
            augmented_image = transformer.augment_image(image)
            # TODO: make this conditional on cuda usage
            augmented_image = augmented_image.to("cuda:0")
            augmented_images.append(augmented_image)
        # Should we send all of them to cuda at teh same time?
        augmented_images = torch.cat(augmented_images, axis=0)
        start = time.time()
        augmented_outputs = self.model(augmented_images,  *args)
        end = time.time()
        print("Took this long: ", start-end)
        if merger.type == 'mean' and not self.ret_all:
            return torch.mean(augmented_outputs, 0).unsqueeze(0)
        pdb.set_trace()
        augmented_outputs = augmented_outputs.view((-1, image.shape[0], 1000))
        if self.output_key is not None:
            augmented_outputs = augmented_outputs[self.output_key]
        deaugmented_output = transformer.deaugment_label(augmented_outputs)
        # TODO: implement batch merge
        for i in range(deaugmented_output.shape[0]):
            merger.append(deaugmented_output[i])
        result = merger.result
        if self.output_key is not None:
            result = {self.output_key: result}
        if self.ret_all:
            return merger.all_x
        return result
