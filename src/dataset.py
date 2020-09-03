import os
from typing import Any, Dict, Union

import cv2
import torch
from torch.utils.data import Dataset

from src import logger


class ImageFolderDataset(Dataset):
    def __init__(self, path: str, transforms: Any = None):
        self._image_names = os.listdir(path)
        self._image_names.sort()
        self._transforms = transforms
        self._path = path
        logger.info(f"dataset from directory \"{path}\" are ready "
                     f"and contains {len(self._image_names)} images")

    def __len__(self) -> int:
        return len(self._image_names)

    def __getitem__(self, index: int) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        image = cv2.imread(os.path.join(self._path, self._image_names[index]),
                           cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self._transforms(image=image)['image']
        return {
            'src_image': image
        }
