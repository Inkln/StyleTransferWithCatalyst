from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio

from catalyst.registry import Criterion

from src import logger
from src.trainable_models import PrunedVggModel
from src.transforms import get_transform


class StyleLossBlock(nn.Module):
    def __init__(self, target: torch.Tensor):
        super().__init__()
        # logger.info(f"GRAM_INPUT: {target.sum():.3f}")
        self.stored_value = None
        self._loss = F.mse_loss
        self.shape = target.shape
        self._target_gram_matrix = nn.Parameter(self.gram_matrix(target).data)
        # logger.info(f"GRAM: {self._target_gram_matrix.sum():.3f}")

    @staticmethod
    def gram_matrix(x: torch.Tensor) -> torch.Tensor:
        (bs, ch, h, w) = x.size()
        f = x.view(bs, ch, w * h)
        f_t = f.transpose(1, 2)
        g = f.bmm(f_t) / (ch * h * w)
        # G = f.mm(f.t()) / (bs * ch * h * w)
        return g

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_gram_matrix = self.gram_matrix(input_tensor)
        result = self._loss(input_gram_matrix, self._target_gram_matrix)
        return result


@Criterion
class StyleLoss(nn.Module):
    def __init__(self,
                 style_filename: str,
                 num_hidden_relu: int = 10,
                 image_size: int = 512,
                 block_indices: Optional[List[int]] = None):
        super().__init__()
        if block_indices is None:
            block_indices = [1, 2, 3, 4, 5, 6, 7, 8]

        self._block_indices = block_indices[:]
        self._losses = nn.ModuleDict()
        self._loaded = False
        logger.info(f"Style loss with indices={str(self._block_indices)} created")
        self.load_style_from_file(style_filename=style_filename,
                                  num_hidden_relu=num_hidden_relu,
                                  image_size=image_size)

    # def load(self, pruned_vgg: PrunedVggModel, input_tensor: torch.Tensor):
    #     logger.info(f"input_tensor.shape={input_tensor.shape}")
    #     style_predictions = pruned_vgg(input_tensor)
    #     for index in self._block_indices:
    #         self._losses[str(index)] = StyleLossBlock(style_predictions[index])
    #     self._loaded = True
    #     logger.info(f"Style loss with indices={str(self._block_indices)} loaded")

    def load_style_from_file(self, style_filename: str, num_hidden_relu: int = 10, image_size: int = 512):
        tmp_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        image = imageio.imread(style_filename)
        transform = get_transform(image_size)
        image = transform(image=image)['image'].unsqueeze(0)
        feature_network = PrunedVggModel(num_relu=num_hidden_relu)
        feature_network.to(tmp_device)
        image = image.to(tmp_device)
        style_predictions = feature_network(image)
        for index in self._block_indices:
            self._losses[str(index)] = StyleLossBlock(style_predictions[index])
        self._loaded = True
        logger.info(f"Style loss with indices={str(self._block_indices)} loaded from file {style_filename} with size {image_size}")

    def forward(self, vgg_predictions: List[torch.Tensor], *args, **kwargs) -> torch.Tensor:
        if not self._loaded:
            raise RuntimeError("Model must be loaded before first forward call")

        result_loss: torch.Tensor = 0
        for index in self._block_indices:
            result_loss += self._losses[str(index)](vgg_predictions[index])
        return result_loss


@Criterion
class ContentLoss(nn.Module):
    def __init__(self,
                 block_indices: Optional[List[int]] = None):
        super().__init__()
        if block_indices is None:
            block_indices = [4]

        self._block_indices = block_indices[:]
        self._loss = F.mse_loss
        logger.info(f"Content loss with indices={str(self._block_indices)} created")

    def forward(self,
                result_predictions: List[torch.Tensor],
                target_predictions: List[torch.Tensor]) -> torch.Tensor:
        result_loss: torch.Tensor = 0
        for index in self._block_indices:
            result_loss += self._loss(result_predictions[index], target_predictions[index])

        return result_loss


@Criterion
class TotalVariationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output_tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        horizontal = torch.abs(output_tensor[:, :, :, 1:] - output_tensor[:, :, :, :-1]).sum()
        vertical = torch.abs(output_tensor[:, :, 1:, :] - output_tensor[:, :, :-1, :]).sum()
        return (horizontal + vertical) / output_tensor.shape[0]
