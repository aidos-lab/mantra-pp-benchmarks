from collections import Counter
from typing import List, Tuple, Dict

import numpy as np
from torch.utils.data import Dataset

from src.metrics.tasks import TaskType


def convert_to_list(imb: Counter) -> Tuple[List[float], List[float]]:
    """Handling of Counter collection"""
    return list(imb.keys()), list(imb.values())


def derive_weights(imbalance: List[float]):
    imbalance = np.asarray(imbalance)
    inversed_imbalance = np.sum(imbalance) / (imbalance * len(imbalance))
    return inversed_imbalance


def normalise_imbalance(imbalance: List[float]) -> List[float]:
    tmp = np.asarray(imbalance)
    return list(tmp / tmp.sum())


def sort_imbalance(
    imbalance: List[float], keys: List
) -> List[float]:
    """
    Sorts the imbalance counters by the index the neural networks intend to predict.
    """
    return  [imbalance[i] for i in range(len(keys))]



def sorted_imbalance_weights(
    dataset: Dataset
) -> List[float]:
    """
    Returns normalised sorted inverse imbalance.

    E.g. if there are 90 samples of class 0 and 10 samples of class 1, it will return [0.1, 0.9]
    """
    imbalance = Counter(dataset.y.tolist())
    print(imbalance)
    imb_keys, imb_values = convert_to_list(imbalance)

    imbalance = sort_imbalance(imb_values, imb_keys)

    inversed_imbalance = derive_weights(imbalance)
    norm_imbalanced = normalise_imbalance(inversed_imbalance)
    return norm_imbalanced

