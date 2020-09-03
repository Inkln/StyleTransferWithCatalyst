import os
import imageio

from PIL import Image
import torch
import torchvision.transforms.transforms as transforms


def image_loader(image_name: str, image_size: int = 512):
    loader = transforms.Compose([
        transforms.Resize(image_size),  # scale imported image
        transforms.ToTensor()]
    )  # transform it into a torch tensor

    image = imageio.imread(image_name)
    image = Image.fromarray(image)
    image = loader(image).unsqueeze(0)
    return image