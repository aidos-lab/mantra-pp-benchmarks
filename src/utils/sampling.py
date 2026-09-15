"""Batch sampler that keeps the snapshots of a Pachner walk together.

The consistency objective reads same-walk pairs off the batch, and
plain shuffling puts ``O(B^2 / n)`` such pairs into a batch of size
``B`` over ``n`` walks, below one for the dataset sizes used here.
"""

from typing import Dict, List, Sequence

import torch
from torch.utils.data import Sampler

MISSING_WALKS_MESSAGE = (
    "trainer.sampler=walks needs a dataset with Pachner walks, but its "
    "entries have no `walk_base`. Set dataset.walk_length > 0."
)


def walks_of(dataset) -> Dict[int, List[int]]:
    """Group the dataset indices by ``walk_base``."""
    walks: Dict[int, List[int]] = {}
    for index, data in enumerate(dataset):
        if "walk_base" not in data:
            raise ValueError(MISSING_WALKS_MESSAGE)
        walks.setdefault(int(data.walk_base), []).append(index)
    return walks


class WalkBatchSampler(Sampler):
    """Yield batches made of whole Pachner walks.

    Every epoch permutes the walks and cuts the permutation into groups
    of ``batch_size // walk_size`` walks, so every same-walk pair is
    seen once per epoch. The last, shorter group is kept.

    Parameters
    ----------
    dataset : Sequence
        Dataset whose entries carry ``walk_base``.
    batch_size : int
        Upper bound on the number of entries per batch; rounded down to
        a multiple of the walk size.
    seed : int
        Seed of the walk permutation; epoch ``e`` uses ``seed + e``.
    """

    def __init__(self, dataset: Sequence, batch_size: int, seed: int = 42):
        self.walks = list(walks_of(dataset).values())
        walk_size = max(len(walk) for walk in self.walks)
        self.walks_per_batch = batch_size // walk_size
        if self.walks_per_batch < 1:
            raise ValueError(
                f"trainer.batch_size={batch_size} is smaller than a walk of "
                f"{walk_size} snapshots"
            )
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        n = len(self.walks)
        return (n + self.walks_per_batch - 1) // self.walks_per_batch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        order = torch.randperm(len(self.walks), generator=generator).tolist()
        for start in range(0, len(order), self.walks_per_batch):
            yield [
                index
                for walk in order[start : start + self.walks_per_batch]
                for index in self.walks[walk]
            ]
