from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import scipy
import torch
from jaxtyping import Float

from general_utils import (
    dict_of_tensors_to_device,
    list_of_tensors_to_device,
)


class NeighborhoodType(Enum):
    UPPER_ADJACENCY = 0
    LOWER_ADJACENCY = 1
    BOUNDARY = 2
    UPPER_HODGE_LAPLACIAN = 3
    LOWER_HODGE_LAPLACIAN = 4
    HODGE_LAPLACIAN = 5


@dataclass
class NeighborhoodMatrixType:
    type: NeighborhoodType
    dimension: int

    def __hash__(self):
        return hash((self.type, self.dimension))


def _other_features_to_device(
    feature: torch.Tensor | list[torch.Tensor] | dict[Any, torch.Tensor],
    device: torch.device,
) -> torch.Tensor | list[torch.Tensor] | dict[Any, torch.Tensor]:
    if isinstance(feature, torch.Tensor):
        return feature.to(device)
    elif isinstance(feature, list):
        return list_of_tensors_to_device(feature, device)
    else:
        return dict_of_tensors_to_device(feature, device)


@dataclass
class SimplicialComplexData:
    pass
