"""Batch sampler that keeps the snapshots of a Pachner walk together.

The consistency objective in :mod:`src.metrics.loss` reads same-walk
pairs off the batch, so a batch is only useful to it if it contains
several snapshots of one walk. Shuffling alone rarely does that: with
``n`` walks the expected number of same-walk pairs in a batch of size
``B`` is ``O(B^2 / n)``, below one for the dataset sizes used here.
This sampler shuffles the *walks* instead and puts every snapshot of a
drawn walk into the same batch.
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
    of ``batch_size // walk_size`` walks, ``walk_size`` being the
    number of snapshots per walk (``walk_length + 1``). A batch thus
    holds all snapshots of its walks and every same-walk pair of the
    dataset is seen exactly once per epoch; an epoch is a pass over the
    data, as with a shuffled loader. The last, shorter group is kept.

    Parameters
    ----------
    dataset : Sequence
        Dataset whose entries carry ``walk_base``, i.e. a
        :class:`mantra.datasets.PachnerWalkDataset` with
        ``walk_length > 0``.
    batch_size : int
        Upper bound on the number of entries per batch. Batches hold
        ``walks_per_batch * walk_size <= batch_size`` entries, so a
        batch size that is not a multiple of the walk size is rounded
        down.
    seed : int
        Seed of the walk permutation; epoch ``e`` uses ``seed + e``.

    Raises
    ------
    ValueError
        If the dataset carries no walk metadata or one walk does not
        fit into a batch.
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
