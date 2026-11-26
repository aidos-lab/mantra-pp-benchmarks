import torch


def feature_stats(train, key, chunk=4096):
    """Per-feature mean and standard deviation of the cached tensor
    attribute `key` over the train split, in the shape of one sample.

    The std is floored at one so that count features varying by less
    than one keep their scale. Accumulated in chunks, so an int16 cache
    of tens of GB is never copied to float as a whole.
    """
    values = train._data[key]
    n = values.shape[0]
    total = torch.zeros(values.shape[1:], dtype=torch.float64)
    squares = torch.zeros_like(total)
    for start in range(0, n, chunk):
        part = values[start : start + chunk].double()
        total += part.sum(dim=0)
        squares += (part * part).sum(dim=0)
    mean = total / n
    var = (squares - n * mean * mean) / max(n - 1, 1)
    return mean.float(), var.clamp_min(0).sqrt().float().clamp_min(1.0)
