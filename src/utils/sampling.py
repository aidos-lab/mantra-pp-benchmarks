"""Batch sampler that fills a batch with informative index pairs.

The Pachner objectives in :mod:`src.metrics.loss` read their pair
structure off the batch, so a batch is only useful to them if it
actually contains pairs of the right kinds. Shuffling alone rarely puts
two snapshots of one walk into the same batch: with ``n`` walks in the
dataset the expected number of same-walk pairs in a batch of size ``B``
is ``O(B^2 / n)``, which is below one for the dataset sizes used here.
This sampler draws the pairs explicitly instead.
"""

import random
from typing import Dict, List, Sequence, Tuple

from torch.utils.data import Sampler

PAIR_KINDS = ("walk", "class", "diff")

MISSING_WALKS_MESSAGE = (
    "trainer.sampler=pachner_pairs needs a dataset with Pachner walks, "
    "but its entries have no `walk_base`/`walk_step`. Set "
    "dataset.walk_length > 0."
)


def _entry_metadata(dataset) -> Tuple[List[int], List[int], List[int]]:
    """Return the per-entry base index, walk step and class label."""
    walk_base, walk_step, label = [], [], []
    for data in dataset:
        if "walk_base" not in data or "walk_step" not in data:
            raise ValueError(MISSING_WALKS_MESSAGE)
        walk_base.append(int(data.walk_base))
        walk_step.append(int(data.walk_step))
        label.append(int(data.y.reshape(-1)[0]))
    return walk_base, walk_step, label


class PachnerPairSampler(Sampler):
    """Yield batches built from same-walk, same-class and cross-class pairs.

    Each batch consists of ``batch_size // 2`` index pairs drawn in the
    proportions given by ``fractions``: two snapshots of one Pachner
    walk, two entries of one class from different walks, and two
    entries of different classes. Indices within a batch are unique, so
    a pair whose partner is already in the batch is redrawn. The pairs
    are drawn afresh in every epoch.

    Parameters
    ----------
    dataset : Sequence
        Dataset whose entries carry ``walk_base``, ``walk_step`` and
        ``y``; that is a
        :class:`mantra.datasets.PachnerWalkDataset` with
        ``walk_length > 0``.
    batch_size : int
        Number of entries per batch. Odd sizes lose the last entry.
    fractions : sequence of float
        Fractions of walk, class and cross-class pairs; must sum to 1.
    seed : int
        Seed of the pair sampling. Every epoch draws from a generator
        seeded with ``seed + epoch``, so the batches are reproducible
        and differ between epochs.

    Raises
    ------
    ValueError
        If the dataset carries no walk metadata, the fractions do not
        sum to one, or the batch holds fewer than two entries.
    """

    def __init__(
        self,
        dataset: Sequence,
        batch_size: int,
        fractions: Sequence[float] = (0.5, 0.25, 0.25),
        seed: int = 42,
    ):
        if batch_size < 2:
            raise ValueError(f"batch_size must be >= 2, got {batch_size}")
        if len(fractions) != len(PAIR_KINDS):
            raise ValueError(
                f"fractions needs {len(PAIR_KINDS)} entries "
                f"({', '.join(PAIR_KINDS)}), got {list(fractions)}"
            )
        if abs(sum(fractions) - 1.0) > 1e-6:
            raise ValueError(f"fractions must sum to 1, got {list(fractions)}")

        self.walk_base, self.walk_step, self.label = _entry_metadata(dataset)
        self.batch_size = batch_size
        self.fractions = [float(f) for f in fractions]
        self.seed = seed
        self.epoch = 0

        self.by_base: Dict[int, List[int]] = {}
        self.by_label: Dict[int, List[int]] = {}
        for index, (base, label) in enumerate(zip(self.walk_base, self.label)):
            self.by_base.setdefault(base, []).append(index)
            self.by_label.setdefault(label, []).append(index)

        self.walk_bases = [
            base for base, group in self.by_base.items() if len(group) > 1
        ]
        if not self.walk_bases:
            raise ValueError(MISSING_WALKS_MESSAGE)
        self.multi_base_labels = [
            label
            for label, group in self.by_label.items()
            if len({self.walk_base[i] for i in group}) > 1
        ]
        self.labels = sorted(self.by_label)

    def __len__(self) -> int:
        return len(self.walk_base) // self.batch_size

    def _pair_plan(self) -> List[str]:
        """Kinds of the pairs of one batch, in the configured fractions."""
        n_pairs = self.batch_size // 2
        counts = [int(round(f * n_pairs)) for f in self.fractions]
        # Rounding can miss or overshoot the batch; the walk pairs are
        # the objectives' primary signal, so they absorb the remainder.
        counts[0] += n_pairs - sum(counts)
        if counts[0] < 0:
            counts = [max(0, count) for count in counts]
        return [
            kind
            for kind, count in zip(PAIR_KINDS, counts)
            for _ in range(count)
        ]

    def _draw(self, kind: str, rng: random.Random) -> Tuple[int, int]:
        """Draw one index pair of the given kind."""
        if kind == "walk":
            group = self.by_base[rng.choice(self.walk_bases)]
            return tuple(rng.sample(group, 2))
        if kind == "class":
            if not self.multi_base_labels:
                return self._draw("walk", rng)
            label = rng.choice(self.multi_base_labels)
            group = self.by_label[label]
            first = rng.choice(group)
            others = [
                index
                for index in group
                if self.walk_base[index] != self.walk_base[first]
            ]
            return first, rng.choice(others)
        if len(self.labels) < 2:
            return self._draw("walk", rng)
        first_label, second_label = rng.sample(self.labels, 2)
        return (
            rng.choice(self.by_label[first_label]),
            rng.choice(self.by_label[second_label]),
        )

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(len(self)):
            batch: List[int] = []
            seen = set()
            for kind in self._pair_plan():
                for _ in range(16):
                    first, second = self._draw(kind, rng)
                    if first not in seen and second not in seen:
                        break
                else:
                    continue
                seen.update((first, second))
                batch.extend((first, second))
            if batch:
                yield batch
