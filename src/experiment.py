from collections import OrderedDict

from torch.utils.data import Dataset
from catalyst.dl import ConfigExperiment
from src import logger

import torch.nn as nn

from .dataset import ImageFolderDataset
from .transforms import get_transform


class Experiment(ConfigExperiment):
    def get_datasets(
        self, stage: str, epoch: int = None, **kwargs,
    ):
        print(f"stage={stage}, epoch={epoch}, kwargs={str(kwargs)}")
        image_size: int = kwargs['image_size']
        return OrderedDict([
            ('train', ImageFolderDataset(transforms=get_transform(image_size=image_size), path=kwargs['path'])),
            # ('valid', ImageFolderDataset(transforms=get_transform(image_size=image_size)))
        ])
