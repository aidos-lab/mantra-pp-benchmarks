import os
import typing
from typing import Literal, Tuple, TypeAlias
from tqdm import tqdm

import torch
from mantra.datasets import ManifoldTriangulations

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.transforms import Compose


from datasets.pairwise import PairwiseSimplicialDS
from metrics.tasks import TaskType, Manifold2Type, Manifold3Type

from sklearn.model_selection import train_test_split

from datasets.transforms import homeo_comp_transforms_2manifold

Mode = Literal["train", "val"]

class BRECPredictionPWDS(InMemoryDataset):
    """
    Wrapper of PairwiseSimplicialDS to extend it with train/test/val split functionality.

    train/test/val splits depend on the task type due to stratification, i.e. that
    a proper split shall maintain the same class imbalance in all splits.
    Since for every task type the class imbalance differs, stratification can only be done dependent
    on the task type.
    """

    def __init__(
        self,
        root: str,
        split: Mode,
        comparison_pair: (
            Tuple[Manifold3Type, Manifold3Type]
            | Tuple[Manifold2Type, Manifold2Type]
        ) = (Manifold2Type.S_2, Manifold2Type.T_2),
        manifold="2",
        version="latest",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        debug = False,
    ):
        self.manifold = manifold
        self.pairwise_ds = PairwiseSimplicialDS(
            root,
            comparison_pair,
            manifold,
            version,
            None,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            debug = debug
        )
        self.transform = transform
        super().__init__(root, transform=None)

        self.load(self._get_processed_path(split))

    @property
    def raw_file_names(self):
        return []

    def download(self) -> None:
        pass

    def _data_filename(self, split: str):
        """
        Parameters
        -----------
            split: ManifoldType
                Type of the second manifold the triangulations represent.
        Returns
        -------
            The name of the data file in .pt format.

        """
        return f"data_brec_format_{split}.pt"

    def _get_processed_path(self, split: str):
        fnames = self.processed_file_names
        idx = 0
        for fname in fnames:
            if fname == self._data_filename(split):
                return self.processed_paths[idx]
            idx += 1
        raise ValueError(
            "Can not find processed data: Unknown config with task type and mode."
        )

    @property
    def processed_file_names(self):
        if self.manifold == "2":
            f_names = [
                self._data_filename(split)
                for split in ["train", "val", "test"]
            ]
        else:
            raise NotImplementedError("TODO")
        return f_names

    def process(self):
        print("---> Preprocessing dataset...)")

        if self.manifold == "3":
            raise NotImplementedError("TODO")

        indices = range(self.pairwise_ds.len())


        if self.transform != None:
            data_list_processed = [self.transform(self.pairwise_ds.get(idx)) for idx in indices]
        else:
            data_list_processed = [self.pairwise_ds.get(idx) for idx in indices]

        
        label_tensor = torch.vstack([data.y for data in data_list_processed])
        train_indices = label_tensor == 0
        val_indices = label_tensor != 0

        #  Save splits
        train_data_list = [data_list_processed[idx] for idx in train_indices]
        val_data_list = [data_list_processed[idx] for idx in val_indices]

        self.save(
            train_data_list, self._get_processed_path("train")
        )
        self.save(
            val_data_list, self._get_processed_path("val")
        )



