
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from torch_geometric.data import Data
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from src.utils.imbalance_handling import sorted_imbalance_weights, compute_target_frequency

def split_dataset(dataset, cfg):
    split_params = cfg.dataset.train_parameters.split

    train_val_index, test_index = train_test_split(
        np.arange(len(dataset)),
        test_size=split_params.test,
        shuffle=True,
        stratify=(
            dataset.y
            if cfg.dataset.use_stratified
            else None
        ),
        random_state=cfg.seed,
    )


    # train val split
    train_index, val_index = train_test_split(
        train_val_index,
        test_size=split_params.val
        / (split_params.train + split_params.val),
        shuffle=True,
        stratify=(
            dataset[train_val_index].y
            if cfg.dataset.use_stratified
            else None
        ),
        random_state=cfg.seed,
    )

    train_dataset = Subset(dataset, train_index)
    val_dataset = Subset(dataset, val_index)
    test_dataset = Subset(dataset, test_index)

    train_loader = DataLoader(train_dataset, batch_size=cfg.trainer.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.trainer.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.trainer.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def filter_low_ocurrence_labels(dataset, n=100):
    keys_to_filter = []
    keys, values = compute_target_frequency(dataset)

    # Compute keys to filter
    for k, v in zip(keys, values):
        if v > n:
            keys_to_filter.append(k)

    mask = torch.zeros(len(dataset), dtype=torch.bool)
    for k in keys_to_filter:
        mask |= (dataset.y == k)
    return dataset[mask]

    


