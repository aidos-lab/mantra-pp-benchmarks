"""Apply a dataset's transforms once instead of on every access.

:class:`torch_geometric.data.InMemoryDataset` runs its ``transform`` on
every ``__getitem__``. Here that pipeline (graph or complex
construction, moment-curve features, feature propagation, label
remapping) is deterministic and dominates the epoch time, so it is
paid once per training run rather than once per epoch and worker.
"""

from typing import Sequence

from torch.utils.data import Dataset
from tqdm import tqdm


class MaterializedDataset(Dataset):
    """A dataset's transformed entries, held in a list.

    Parameters
    ----------
    dataset : Sequence
        Dataset whose ``__getitem__`` applies the transforms.
    desc : str
        Progress-bar label.
    """

    def __init__(self, dataset: Sequence, desc: str = "Materializing"):
        self.entries = [
            dataset[i] for i in tqdm(range(len(dataset)), desc=desc)
        ]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index):
        return self.entries[index]
