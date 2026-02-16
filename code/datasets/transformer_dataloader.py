from typing import Sequence, Any, Optional

import numpy as np
import scipy
import torch
from jaxtyping import Float, Int
from torch.utils.data import DataLoader

from datasets.utils import concat_tensors
from datasets.utils import torch_sparse_to_scipy_sparse
from models.cells.transformer.DataTypes import (
    NeighborhoodMatrixType,
    NeighborhoodType,
)




def collate_other_features_type_dependent(
    feature_list: list[
        Float[torch.Tensor, "..."]
        | dict[Any, Float[torch.Tensor, "..."]]
        | list[Float[torch.Tensor, "..."]]
    ]
) -> (
    Float[torch.Tensor, "..."]
    | dict[Any, Float[torch.Tensor, "..."]]
    | list[Float[torch.Tensor, "..."]]
):
    data_example = feature_list[0]
    collated_subfeatures = None
    if isinstance(data_example, dict):
        collated_subfeatures = dict()
        feature_names = set(
            [key for example in feature_list for key in example.keys()]
        )
        for key in feature_names:
            collated_subfeatures[key] = torch.cat(
                [example[key] for example in feature_list if key in example],
                dim=0,
            )
    if isinstance(data_example, torch.Tensor):
        collated_subfeatures = concat_tensors(feature_list, dim=0)
    if isinstance(data_example, list):
        # We assume that all the lists have the same length.
        collated_subfeatures = [
            torch.cat([example[i] for example in feature_list], dim=0)
            for i in range(len(data_example))
        ]
    if collated_subfeatures is None:
        raise ValueError("Unknown other features type")
    return collated_subfeatures

