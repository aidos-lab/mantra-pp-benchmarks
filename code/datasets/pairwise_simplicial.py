import os
from collections import Counter
from typing import Callable, List, Tuple

from lightning import LightningDataModule
from torch_geometric.loader import DataLoader as DataLoaderGeometric
from torch_geometric.transforms import Compose

from datasets.dataset_types import DatasetType, filter_nameless
from datasets.transforms import (
    BarycentricSubdivisionTransform,
    SimplicialComplexTransform,
)
from metrics.tasks import Manifold2Type, Manifold3Type, TaskType
from .homeo_pred_brec import BRECPredictionPWDS


def unique_counts(input_list: List[str]) -> Counter:
    return Counter(input_list)


def barycentric_subdivision_transform(
    num_barycentric_subdivisions: int = 1,
) -> Compose:
    return Compose(
        [
            SimplicialComplexTransform(),
            BarycentricSubdivisionTransform(num_barycentric_subdivisions),
        ]
    )


class PairwiseSimplicialDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        split: str = 'train',
        transform: Compose | None = None,
        use_stratified: bool = False,
        comparison_pair: Tuple[Manifold3Type, Manifold3Type] | Tuple[Manifold2Type, Manifold2Type] = (Manifold2Type.S_2, Manifold2Type.T_2),
        batch_size: int = 128,
        seed: int = 2024,
        dataloader_builder: Callable = DataLoaderGeometric,
        ds_type: DatasetType = DatasetType.FULL_2D,
        num_barycentric_subdivisions: int = 0,
    ):
        super().__init__()
        self.split = split
        self.ds_type = ds_type
        self.data_dir = data_dir
        self.transform = transform
        self.use_stratified = use_stratified
        self.comparison_pair = comparison_pair
        self.stratified = None
        self.batch_size = batch_size
        self.seed = seed
        self.dataloader_builder = dataloader_builder
        self.split = (0.6, 0.2, 0.2)
        self.num_barycentric_subdivisions = num_barycentric_subdivisions

    def get_ds_root_dir(self) -> str:
        return os.path.join(
            self.data_dir,
            self.ds_type.name.lower(),
            f"{self.num_barycentric_subdivisions}",
        )

    def get_ds(self, mode: str = "train") -> BRECPredictionPWDS:
        if self.ds_type == DatasetType.FULL_2D:
            manifold = "2"
            pre_filter = None
        elif self.ds_type == DatasetType.FULL_3D:
            manifold = "3"
            pre_filter = None
        elif self.ds_type == DatasetType.NO_NAMELESS_2D:
            manifold = "2"
            pre_filter = filter_nameless
        elif self.ds_type == DatasetType.PAIRWISE_HOMEO_2D:
            manifold = "2"
            pre_filter = None
        elif self.ds_type == DatasetType.PAIRWISE_HOMEO_3D:
            manifold = "3"
            pre_filter = None
        else:
            raise ValueError(f"Unknown dataset type {self.ds_type}")

        barycentric_subdivision_pre_tr = (
            barycentric_subdivision_transform(
                self.num_barycentric_subdivisions
            )
            if self.num_barycentric_subdivisions > 0
            else None
        )

        return BRECPredictionPWDS(
            root=self.get_ds_root_dir(),
            split=self.split,
            comparison_pair=self.comparison_pair,
            manifold=manifold,
            transform=self.transform,
            pre_transform=None,
            pre_filter=None,
        )

    def prepare_data(self) -> None:
        self.get_ds()

    def class_imbalance_statistics(self) -> Counter:
        dataset = self.get_ds()

        statistics = unique_counts(dataset.y)
        return statistics

    def setup(self, stage=None):
        self.train_ds = self.get_ds("train")
        self.val_ds = self.get_ds("val")

    def train_dataloader(self):
        return self.dataloader_builder(
            self.train_ds, batch_size=self.batch_size, num_workers=8
        )

    def val_dataloader(self):
        return self.dataloader_builder(
            self.val_ds, batch_size=self.batch_size, num_workers=8
        )


if __name__ == "__main__":
    dm = PairwiseSimplicialDataModule()
