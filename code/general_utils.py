from typing import Any

import numpy as np
import scipy
import torch



def dict_of_tensors_to_device(
    tensor_dict: dict[Any, torch.Tensor], device: torch.device
) -> dict[Any, torch.Tensor]:
    return {key: value.to(device) for key, value in tensor_dict.items()}


def list_of_tensors_to_device(
    tensor_list: list[torch.Tensor], device: torch.device
) -> list[torch.Tensor]:
    return [x.to(device) for x in tensor_list]
