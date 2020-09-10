import os
from typing import Optional

import imageio

import torch
import torch.jit as jit

from src import logger
from src.trainable_models import ImageTransformer
from src.transforms import get_transform, post_transform


def get_traced_name(checkpoint_path: str, image_size: int = 512) -> str:
    return f"{checkpoint_path}.{image_size}.traced"


def trace(checkpoint_path: str,
          image_size: int = 512,
          device_name: str = 'cpu') -> None:
    logger.info(f'Model {checkpoint_path} for image_size={image_size} will be traced')
    model = ImageTransformer()
    device = torch.device(device_name)

    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
    model.eval()

    input_tensor = torch.randn(1, 3, image_size, image_size)
    traced_module = jit.trace_module(mod=model,
                                     inputs={'inference': input_tensor})

    traced_checkpoint_path = get_traced_name(checkpoint_path, image_size)
    torch.jit.save(traced_module, traced_checkpoint_path)
    logger.info(f'Traced model saved to {traced_checkpoint_path}')


def infer(image_path: str,
          result_path: str,
          checkpoint_path: str,
          image_size: int = 512,
          device_name: Optional[str] = None):
    device: torch.device
    if device_name is not None:
        device = torch.device(device_name)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    logger.info(f'device selected as {device}')

    model: torch.jit.ScriptModule
    traced_checkpoint_path = get_traced_name(checkpoint_path, image_size)
    #if not os.path.exists(traced_checkpoint_path):
    trace(checkpoint_path, image_size=image_size, device_name=device_name)

    logger.info(f'Traced model loaded from {traced_checkpoint_path}')
    model = torch.jit.load(traced_checkpoint_path)
    model = model.to(device)

    transform = get_transform(image_size=image_size)
    image = imageio.imread(image_path)
    image_shape = image.shape
    image_tensor = transform(image=image)['image'].unsqueeze(0).to(device)

    styled_image_tensor = model.inference(image_tensor)
    styled_image = post_transform(styled_image_tensor[0])

    if image_shape[0] > image_shape[1]:
        delta = int(round((1 - image_shape[1] / image_shape[0]) / 2 * image_size))
        styled_image = styled_image[:, delta:-delta]
    else:
        delta = int(round((1 - image_shape[0] / image_shape[1]) / 2 * image_size))
        styled_image = styled_image[delta:-delta, :]

    imageio.imwrite(result_path, styled_image)


if __name__ == "__main__":
    infer(image_path='https://i.trbna.com/preset/wysiwyg/7/bc/094eaab2b11e9aa1fd76da6667122.jpeg',
          result_path='result.png',
          checkpoint_path='logs/checkpoints/last.pth',
          image_size=512,
          device_name='cuda:0')
