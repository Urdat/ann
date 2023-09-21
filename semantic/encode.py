from os import PathLike
from torch import Tensor
from torch.nn import Module, Sequential
from typing import Callable, Sequence
from torchvision.models import EfficientNet_B7_Weights
from torchvision.transforms import Compose, Normalize, Resize

from torchvision.io import read_image
from torchvision.models import efficientnet_b7
from tqdm import tqdm

import torch

__all__ = (
    'Encoder',
    'default_encoder'
)


def ensure_strings(
        paths: Sequence[PathLike | str] | PathLike | str
) -> list[str]:
    if isinstance(paths, PathLike | str):
        return [str(paths)]
    return [str(path) for path in paths]


class Encoder:

    def __init__(self, module: Module, *, transform: Callable = None) -> None:
        self.module: Module = module
        self.transform: Callable = transform or Compose([])

    @torch.no_grad()
    def encode_file(
            self,
            paths: list[str] | str,
            *,
            verbose: bool = False
    ) -> Tensor:
        paths: list[str] = ensure_strings(paths)
        if verbose is True:
            paths: tqdm = tqdm(paths, desc='Encoding')
        tensors: list[Tensor] = []
        for path in paths:
            tensor: Tensor = read_image(path).float().unsqueeze(0)
            tensor: Tensor = self.transform(tensor)
            tensor: Tensor = self.module(tensor).flatten()
            tensors.append(tensor)
        return torch.stack(tensors)


def default_encoder(
        transform: Callable = None,
        *,
        progress: bool = False
) -> Encoder:
    module = efficientnet_b7(
        weights=EfficientNet_B7_Weights.IMAGENET1K_V1,
        progress=progress
    ).eval().cpu()
    transform = transform or Compose([
        Normalize([127.5] * 3, [127.5] * 3),
        Resize(256, antialias=True)
    ])
    return Encoder(
        module=Sequential(
            module.features,
            module.avgpool
        ),
        transform=transform
    )
