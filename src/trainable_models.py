from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torchvision.models

from catalyst.registry import Model

from src import logger


class Normalization(nn.Module):
    def __init__(self):
        super(Normalization, self).__init__()
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1))
        self.std = nn.Parameter(torch.tensor([0.329, 0.224, 0.225]).view(-1, 1, 1))

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class PrunedVggModel(nn.Module):
    def __init__(self, num_relu: int):
        super().__init__()
        self._num_relu = num_relu

        self._model_blocks = nn.ModuleList([Normalization()])
        vgg_features = torchvision.models.vgg19(pretrained=True).features.eval()
        current_block = nn.Sequential()
        for index, layer in enumerate(vgg_features):
            logger.info(f"Add module {type(layer).__name__}.{index}, "
                         f"weight.sum()={layer.weight.sum() if isinstance(layer, nn.Conv2d) else 0.0:.03f}")
            if isinstance(layer, nn.ReLU):
                current_block.add_module(str(index), nn.ReLU(inplace=False))
                self._model_blocks.append(current_block)
                logger.info("Flush current block")
                if len(self._model_blocks) > self._num_relu:
                    break

                current_block = nn.Sequential()
            else:
                current_block.add_module(str(index), layer)
        for current_block in self._model_blocks:
            for param in current_block.parameters():
                param.requires_grad_(False)

        logger.info(f"Pruned VGG model ready")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        output = []
        for block in self._model_blocks:
            x = block(x)
            output.append(x)
        return output


class ConvLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int):
        super().__init__()
        self._conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride)
        self._pad = nn.ReflectionPad2d(padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad(x)
        x = self._conv(x)
        return x


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int, upsample: Optional[int] = None):
        super().__init__()
        self._upsample = None if upsample is None else nn.UpsamplingNearest2d(scale_factor=upsample)
        self._conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride)
        self._pad = nn.ReflectionPad2d(padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._upsample is not None:
            x = self._upsample(x)
        x = self._pad(x)
        x = self._conv(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super(ResidualBlock, self).__init__()
        self._conv1 = ConvLayer(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, stride=1)
        self._in1 = nn.InstanceNorm2d(num_features=channels, affine=True)
        self._relu = nn.ReLU()
        self._conv2 = ConvLayer(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, stride=1)
        self._in2 = nn.InstanceNorm2d(num_features=channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self._relu(self._in1(self._conv1(x)))
        out = self._in2(self._conv2(out))
        out = out + residual
        out = self._relu(out)
        return out


@Model
class ImageTransformer(nn.Module):
    def __init__(self, num_hidden_relu: int = 15):
        super().__init__()

        self._encoder = nn.Sequential(
            ConvLayer(in_channels=3, out_channels=32, kernel_size=9, stride=1),
            nn.InstanceNorm2d(num_features=32, affine=True),
            nn.ReLU(),

            ConvLayer(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.InstanceNorm2d(num_features=64, affine=True),
            nn.ReLU(),

            ConvLayer(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.InstanceNorm2d(num_features=128, affine=True),
            nn.ReLU()
        )

        self._residual = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )

        self._decoder = nn.Sequential(
            UpsampleConvLayer(in_channels=128, out_channels=64, kernel_size=3, stride=1, upsample=2),
            nn.InstanceNorm2d(num_features=64, affine=True),
            nn.ReLU(),

            UpsampleConvLayer(in_channels=64, out_channels=32, kernel_size=3, stride=1, upsample=2),
            nn.InstanceNorm2d(num_features=32, affine=True),
            nn.ReLU(),

            UpsampleConvLayer(in_channels=32, out_channels=3, kernel_size=9, stride=1, upsample=None),
        )

        self._vgg = PrunedVggModel(num_relu=num_hidden_relu)

    def forward(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = input_tensor

        x = self._encoder(x)
        x = self._residual(x)
        x = self._decoder(x)

        vgg_generated_features = self._vgg(x)
        vgg_origin_features = self._vgg(input_tensor)

        return {
            'generated_image': x,
            'vgg_generated_features': vgg_generated_features,
            'vgg_origin_features': vgg_origin_features
        }

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        x = self._encoder(x)
        x = self._residual(x)
        x = self._decoder(x)

        return x