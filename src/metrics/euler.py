"""Accuracy at the level of the Euler characteristic.

On the 2D name task the nine classes fall into seven values of chi
(the torus and the Klein bottle share chi = 0, #^2 T^2 and #^4 RP^2
share chi = -2), and a model that knows chi and nothing else scores at
most 7/9 balanced accuracy. ``chi_balanced_accuracy`` collapses
predictions and targets to their chi and asks how often the chi is
right, so the two questions "does the model know chi" and "does it
know orientability" can be told apart.
"""

import re

import torch

from mantra.transforms import NAME_TO_CLASS_2M


def chi_of_surface(name: str) -> int:
    """Euler characteristic of a closed surface from its MANTRA name."""
    if name == "S^2":
        return 2
    if name == "Klein bottle":
        return 0
    match = re.fullmatch(r"(?:#\^(\d+) )?(T\^2|RP\^2)", name)
    if match is None:
        raise ValueError(f"Not a 2-manifold name: {name!r}")
    count = int(match.group(1) or 1)
    return 2 - 2 * count if match.group(2) == "T^2" else 2 - count


def chi_of_classes(label_to_index: dict) -> torch.Tensor:
    """chi of every contiguous class index of a 2D name run."""
    name_of_label = {index: name for name, index in NAME_TO_CLASS_2M.items()}
    chi = torch.zeros(len(label_to_index), dtype=torch.long)
    for label, index in label_to_index.items():
        chi[index] = chi_of_surface(name_of_label[int(label)])
    return chi


def chi_balanced_accuracy(y: torch.Tensor, y_hat: torch.Tensor, chi: torch.Tensor) -> float:
    """Macro accuracy over the chi values present in ``y``."""
    chi = chi.to(y.device)
    true, pred = chi[y], chi[y_hat]
    per_value = [(pred[true == c] == c).float().mean() for c in true.unique()]
    return torch.stack(per_value).mean().item()
