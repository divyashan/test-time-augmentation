from functools import partial
from typing import Optional, List, Union, Tuple
from . import functional as F
from .base import DualTransform, ImageOnlyTransform

from numpy import random
from .augmentation_transforms_hpy import get_all_transform_fs 
from .utils import pil_wrap_imgs, pil_unwrap_imgs
import pdb

class ColorJitter(DualTransform):
    identity_param = False

    def __init__(self, brightness: float = .1, contrast: float = 0, saturation: float = 0,  hue: float = 0):
        # TODO: reverse tehse
        super().__init__("apply", [False, True])
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
         
    def apply_aug_image(self, image, apply=False, **kwargs):
        if apply:
            brightness_factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            contrast_factor= random.uniform(max(0, 1 - self.contrast), 1 + self.contrast) 
            saturation_factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation) 
            hue_factor = random.uniform(-self.hue, self.hue) 
            return F.jitter(image, brightness_factor, contrast_factor, saturation_factor, hue_factor) 
        return image

    def apply_deaug_mask(self, mask, apply=False, **kwargs):
        raise ValueError("`ColorJitter` augmentation is not suitable for mask!")

    def apply_deaug_label(self, label, apply=False, **kwargs):
        return label

class Rot(ImageOnlyTransform):
    def __init__(self, angles):
        angles = [0] + angles 
        rot_functions = (
            partial(F.rotate, angle=x) for x in angles
        )
        super().__init__("rot_fn", rot_functions)
         
    def apply_aug_image(self, image, rot_fn=None, **kwargs):
        return rot_fn(image)

    def apply_deaug_mask(self, mask, apply=False, **kwargs):
        raise ValueError("`Rotate` augmentation is not suitable for mask!")

    def apply_deaug_label(self, label, apply=False, **kwargs):
        return label

class HorizontalFlip(DualTransform):
    """Flip images horizontally (left->right)"""

    identity_param = False

    def __init__(self):
        super().__init__("apply", [False, True])

    def apply_aug_image(self, image, apply=False, **kwargs):
        if apply:
            image = F.hflip(image)
        return image

    def apply_deaug_mask(self, mask, apply=False, **kwargs):
        if apply:
            mask = F.hflip(mask)
        return mask

    def apply_deaug_label(self, label, apply=False, **kwargs):
        return label


class VerticalFlip(DualTransform):
    """Flip images vertically (up->down)"""

    identity_param = False

    def __init__(self):
        super().__init__("apply", [False, True])

    def apply_aug_image(self, image, apply=False, **kwargs):
        if apply:
            image = F.vflip(image)
        return image

    def apply_deaug_mask(self, mask, apply=False, **kwargs):
        if apply:
            mask = F.vflip(mask)
        return mask

    def apply_deaug_label(self, label, apply=False, **kwargs):
        return label


class Rotate90(DualTransform):
    """Rotate images 0/90/180/270 degrees

    Args:
        angles (list): angles to rotate images
    """

    identity_param = 0

    def __init__(self, angles: List[int]):
        if self.identity_param not in angles:
            angles = [self.identity_param] + list(angles)

        super().__init__("angle", angles)

    def apply_aug_image(self, image, angle=0, **kwargs):
        k = angle // 90 if angle >= 0 else (angle + 360) // 90
        return F.rot90(image, k)

    def apply_deaug_mask(self, mask, angle=0, **kwargs):
        return self.apply_aug_image(mask, -angle)

    def apply_deaug_label(self, label, angle=0, **kwargs):
        return label


class Scale(DualTransform):
    """Scale images

    Args:
        scales (List[Union[int, float]]): scale factors for spatial image dimensions
        interpolation (str): one of "nearest"/"lenear" (see more in torch.nn.interpolate)
        align_corners (bool): see more in torch.nn.interpolate
    """

    identity_param = 1

    def __init__(
        self,
        scales: List[Union[int, float]],
        interpolation: str = "nearest",
        align_corners: Optional[bool] = None,
    ):
        if self.identity_param not in scales:
            scales = [self.identity_param] + list(scales)
        self.interpolation = interpolation
        self.align_corners = align_corners

        super().__init__("scale", scales)

    def apply_aug_image(self, image, scale=1, **kwargs):
        if scale != self.identity_param:
            image = F.scale(
                image,
                scale,
                interpolation=self.interpolation,
                align_corners=self.align_corners,
            )
        return image

    def apply_deaug_mask(self, mask, scale=1, **kwargs):
        if scale != self.identity_param:
            mask = F.scale(
                mask,
                1 / scale,
                interpolation=self.interpolation,
                align_corners=self.align_corners,
            )
        return mask

    def apply_deaug_label(self, label, scale=1, **kwargs):
        return label


class Resize(DualTransform):
    """Resize images

    Args:
        sizes (List[Tuple[int, int]): scale factors for spatial image dimensions
        original_size Tuple(int, int): optional, image original size for deaugmenting mask
        interpolation (str): one of "nearest"/"lenear" (see more in torch.nn.interpolate)
        align_corners (bool): see more in torch.nn.interpolate
    """

    def __init__(
        self,
        sizes: List[Tuple[int, int]],
        original_size: Tuple[int, int] = None,
        interpolation: str = "nearest",
        align_corners: Optional[bool] = None,
    ):
        if original_size is not None and original_size not in sizes:
            sizes = [original_size] + list(sizes)
        self.interpolation = interpolation
        self.align_corners = align_corners
        self.original_size = original_size

        super().__init__("size", sizes)

    def apply_aug_image(self, image, size, **kwargs):
        if size != self.original_size:
            image = F.resize(
                image,
                size,
                interpolation=self.interpolation,
                align_corners=self.align_corners,
            )
        return image

    def apply_deaug_mask(self, mask, size, **kwargs):
        if self.original_size is None:
            raise ValueError(
                "Provide original image size to make mask backward transformation"
            )
        if size != self.original_size:
            mask = F.resize(
                mask,
                self.original_size,
                interpolation=self.interpolation,
                align_corners=self.align_corners,
            )
        return mask

    def apply_deaug_label(self, label, size=1, **kwargs):
        return label


class Add(ImageOnlyTransform):
    """Add value to images

    Args:
        values (List[float]): values to add to each pixel
    """

    identity_param = 0

    def __init__(self, values: List[float]):

        if self.identity_param not in values:
            values = [self.identity_param] + list(values)
        super().__init__("value", values)

    def apply_aug_image(self, image, value=0, **kwargs):
        if value != self.identity_param:
            image = F.add(image, value)
        return image


class Multiply(ImageOnlyTransform):
    """Multiply images by factor

    Args:
        factors (List[float]): factor to multiply each pixel by
    """

    identity_param = 1

    def __init__(self, factors: List[float]):
        if self.identity_param not in factors:
            factors = [self.identity_param] + list(factors)
        super().__init__("factor", factors)

    def apply_aug_image(self, image, factor=1, **kwargs):
        if factor != self.identity_param:
            image = F.multiply(image, factor)
        return image


class FiveCrops(ImageOnlyTransform):
    """Makes 4 crops for each corner + center crop

    Args:
        crop_height (int): crop height in pixels
        crop_width (int): crop width in pixels 
    """

    def __init__(self, crop_height, crop_width):
        crop_functions = (
            partial(F.crop_lt, crop_h=crop_height, crop_w=crop_width),
            partial(F.crop_lb, crop_h=crop_height, crop_w=crop_width),
            partial(F.crop_rb, crop_h=crop_height, crop_w=crop_width),
            partial(F.crop_rt, crop_h=crop_height, crop_w=crop_width),
            partial(F.crop_c, crop_h=crop_height, crop_w=crop_width),
        )
        super().__init__("crop_fn", crop_functions)

    def apply_aug_image(self, image, crop_fn=None, **kwargs):
        return crop_fn(image)

    def apply_deaug_mask(self, mask, **kwargs):
        raise ValueError("`FiveCrop` augmentation is not suitable for mask!")


class ModifiedFiveCrops(ImageOnlyTransform):
    """Makes 4 crops for each corner + center crop

    Args:
        crop_height (int): crop height in pixels
        crop_width (int): crop width in pixels 
    """

    def __init__(self, crop_height, crop_width):
        crop_functions = (
            partial(F.crop_orig, crop_h=crop_height, crop_w=crop_width),
            partial(F.crop_lt, crop_h=crop_height, crop_w=crop_width),
            partial(F.crop_lb, crop_h=crop_height, crop_w=crop_width),
            partial(F.crop_rb, crop_h=crop_height, crop_w=crop_width),
            partial(F.crop_rt, crop_h=crop_height, crop_w=crop_width),
            partial(F.crop_c, crop_h=crop_height, crop_w=crop_width),
        )
        super().__init__("crop_fn", crop_functions)

    def apply_aug_image(self, image, crop_fn=None, **kwargs):
        if crop_fn == None:
            return image
        return crop_fn(image)

    def apply_deaug_mask(self, mask, **kwargs):
        raise ValueError("`FiveCrop` augmentation is not suitable for mask!")


class AllPIL(ImageOnlyTransform):
    """Makes 4 crops for each corner + center crop

    Args:
        crop_height (int): crop height in pixels
        crop_width (int): crop width in pixels 
    """

    def __init__(self, im_size, dataset):
        all_functions = get_all_transform_fs(im_size) 
        # Here we enumerate all of the partial functions that PIL offers...based on AutoAugment paper
        super().__init__("aug_fn", all_functions)
        self.crop_h= im_size
        self.crop_w = im_size 
        self.dataset = dataset
    def apply_aug_image(self, image, aug_fn=None, **kwargs):
        if aug_fn== None:
            return image
        image = F.center_crop(image, self.crop_h, self.crop_w)
        images_pil = pil_wrap_imgs(image, self.dataset)
        aug_images_pil = [aug_fn(x) for x in images_pil]
        aug_images_tensor = pil_unwrap_imgs(aug_images_pil, self.dataset)
        return aug_images_tensor 

    def apply_deaug_mask(self, mask, **kwargs):
        raise ValueError("`FiveCrop` augmentation is not suitable for mask!")
