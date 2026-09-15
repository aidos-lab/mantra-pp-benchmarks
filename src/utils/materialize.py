"""Apply a dataset's transforms once instead of on every access.

:class:`torch_geometric.data.InMemoryDataset` runs its ``transform`` on
every ``__getitem__``. Here that pipeline (graph or complex
construction, moment-curve features, feature propagation, label
remapping) is deterministic and dominates the epoch time, so it is
paid once per training run rather than once per epoch and worker.
"""

from typing import Iterable, Sequence

from torch.utils.data import Dataset
from tqdm import tqdm

# Raw payload the transforms consume but nothing downstream reads; a
# Python list of lists per entry, about half of an entry's memory.
DROP_KEYS = ("triangulation", "simplex_trie")


class MaterializedDataset(Dataset):
    """A dataset's transformed entries, held in a list.

    Parameters
    ----------
    dataset : Sequence
        Dataset whose ``__getitem__`` applies the transforms.
    desc : str
        Progress-bar label.
    drop : iterable of str
        Attributes removed from every entry after the transforms ran.

    Notes
    -----
    Use with ``num_workers=0``: forked loader workers would each end up
    with their own copy of the list (refcount writes defeat
    copy-on-write), which multiplies the memory by the worker count.
    """

    def __init__(
        self,
        dataset: Sequence,
        desc: str = "Materializing",
        drop: Iterable[str] = DROP_KEYS,
    ):
        drop = tuple(drop)
        self.entries = []
        for i in tqdm(range(len(dataset)), desc=desc):
            data = dataset[i]
            for key in drop:
                if key in data:
                    del data[key]
            self.entries.append(data)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index):
        return self.entries[index]
