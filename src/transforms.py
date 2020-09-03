import cv2
import torch
import numpy as np
import albumentations as albu
from albumentations.pytorch import ToTensorV2


def get_transform(image_size: int = 512):
    transform = albu.Compose([
        albu.LongestMaxSize(max_size=image_size),
        albu.PadIfNeeded(min_height=image_size, min_width=image_size, value=0,
                         border_mode=cv2.BORDER_CONSTANT),
        albu.Normalize(mean=0, std=1),
        ToTensorV2()
    ])
    return transform


def post_transform(image_tensor: torch.Tensor) -> np.array:
    image = image_tensor.cpu().detach().numpy()
    image = image.transpose(1, 2, 0).clip(0, 1)
    image *= 255
    image = image.astype(np.uint8)
    return image
