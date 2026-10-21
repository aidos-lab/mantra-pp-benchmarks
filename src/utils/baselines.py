"""Trivial baselines and target statistics read from the raw split columns.

`InMemoryDataset` keeps every split's raw attributes concatenated in
`_data`, so none of this needs a pass over the transformed samples.
"""

def target_values(cfg, dataset):
    """Raw target column of `dataset` as one flat float tensor."""
    return dataset._data[cfg.task.target].reshape(-1).float()


def feature_stats(train, key):
    """Per-feature mean and standard deviation of a cached tensor
    attribute such as `ect`, over the train split; the std is floored at
    one so that features varying by less than one count keep their scale."""
    values = train._data[key].float()
    return values.mean(dim=0), values.std(dim=0).clamp_min(1.0)


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
