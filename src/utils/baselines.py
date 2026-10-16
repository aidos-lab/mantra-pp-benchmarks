"""Trivial baselines and target statistics read from the raw split columns.

`InMemoryDataset` keeps every split's raw attributes concatenated in
`_data`, so none of this needs a pass over the transformed samples.
"""

def target_values(cfg, dataset):
    """Raw target column of `dataset` as one flat float tensor."""
    return dataset._data[cfg.task.target].reshape(-1).float()


def regression_target_stats(cfg, train):
    """Mean and standard deviation of the train-split regression target."""
    values = target_values(cfg, train)
    return values.mean().item(), values.std().clamp_min(1e-6).item()


def trivial_baselines(cfg, train, test):
    """Test-split scores of the majority class or the train median."""
    y_train, y_test = target_values(cfg, train), target_values(cfg, test)
    if cfg.task.task_type == "classification":
        values, counts = y_train.unique(return_counts=True)
        majority = values[counts.argmax()]
        return {
            "baseline/test_accuracy": (y_test == majority).float().mean().item(),
            "baseline/test_balanced_accuracy": 1.0 / len(values),
        }
    median = y_train.median()
    return {
        "baseline/test_mae": (y_test - median).abs().mean().item(),
        "baseline/test_mse": ((y_test - y_train.mean()) ** 2).mean().item(),
        "baseline/test_rounded_accuracy": (y_test == median.round()).float().mean().item(),
    }
