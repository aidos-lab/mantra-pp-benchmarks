"""Quality measures of the learned embedding.

The Pachner objectives shape the embedding rather than the classifier
head, so accuracy alone does not say whether they worked. Measured
here: does embedding distance track the number of Pachner moves
between two snapshots of a walk, and does a k-NN classifier on the
training embeddings recover the homeomorphism type.
"""

from typing import Optional

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier

MAX_PAIRS = 20000


@torch.no_grad()
def collect_embeddings(model, loader, cfg):
    """Return ``(embedding, y, walk_base, walk_step)`` of a loader as arrays.

    The walk arrays are None for a dataset without Pachner walks.
    """
    model.eval()
    embeddings, labels, bases, steps = [], [], [], []
    has_walks = None

    for batch in loader:
        batch = batch.to(cfg.trainer.device)
        _, embedding = model(batch, return_embedding=True)
        embeddings.append(embedding.detach().cpu())
        labels.append(batch.y.detach().cpu().reshape(-1))

        walks = "walk_base" in batch and "walk_step" in batch
        if has_walks is None:
            has_walks = walks
        elif walks != has_walks:
            raise ValueError(
                "Some batches of this split carry walk metadata and "
                "others do not; the split is inconsistent."
            )
        if walks:
            bases.append(batch.walk_base.detach().cpu().reshape(-1))
            steps.append(batch.walk_step.detach().cpu().reshape(-1))

    embedding = torch.cat(embeddings).numpy()
    y = torch.cat(labels).numpy()
    if not has_walks:
        return embedding, y, None, None
    return embedding, y, torch.cat(bases).numpy(), torch.cat(steps).numpy()


def _distance_walk_correlation(
    embedding: np.ndarray,
    walk_base: Optional[np.ndarray],
    walk_step: Optional[np.ndarray],
    moves_per_step: int,
    rng: np.random.Generator,
) -> Optional[float]:
    """Spearman correlation of embedding distance and walk distance."""
    if walk_base is None:
        return None

    first, second = [], []
    for base in np.unique(walk_base):
        group = np.flatnonzero(walk_base == base)
        if len(group) < 2:
            continue
        rows, cols = np.triu_indices(len(group), k=1)
        first.append(group[rows])
        second.append(group[cols])
    if not first:
        return None

    first = np.concatenate(first)
    second = np.concatenate(second)
    if len(first) > MAX_PAIRS:
        keep = rng.choice(len(first), size=MAX_PAIRS, replace=False)
        first, second = first[keep], second[keep]

    distance = np.linalg.norm(embedding[first] - embedding[second], axis=-1)
    moves = np.abs(walk_step[first] - walk_step[second]) * moves_per_step
    if len(np.unique(moves)) < 2 or len(np.unique(distance)) < 2:
        return None
    return float(spearmanr(distance, moves).statistic)


def evaluate_embeddings(model, train_loader, eval_loaders, cfg):
    """Measure how well the embedding encodes walks and classes.

    Parameters
    ----------
    model : nn.Module
        Trained model whose `forward` takes `return_embedding`.
    train_loader : DataLoader
        Loader whose embeddings index the k-NN classifier.
    eval_loaders : dict of str to DataLoader
        Splits to evaluate, keyed by their metric prefix.
    cfg : RootConfig
        Composed config; the device, the seed and
        `dataset.moves_per_step` are read.

    Returns
    -------
    dict of str to float
        For every split: `<prefix>/emb_spearman` (rank correlation of
        embedding distance with the number of moves between two
        snapshots of one walk) and `<prefix>/emb_knn_bal_acc`. Measures
        a split cannot support (no walks, a single class) are left out.
    """
    rng = np.random.default_rng(cfg.seed)
    train_embedding, train_y, _, _ = collect_embeddings(
        model, train_loader, cfg
    )
    knn = None
    if len(np.unique(train_y)) >= 2:
        knn = KNeighborsClassifier(n_neighbors=min(5, len(train_y)))
        knn.fit(train_embedding, train_y)

    results = {}
    for prefix, loader in eval_loaders.items():
        embedding, y, walk_base, walk_step = collect_embeddings(
            model, loader, cfg
        )
        spearman = _distance_walk_correlation(
            embedding, walk_base, walk_step, cfg.dataset.moves_per_step, rng
        )
        if spearman is not None:
            results[f"{prefix}/emb_spearman"] = spearman
        if knn is not None:
            results[f"{prefix}/emb_knn_bal_acc"] = float(
                balanced_accuracy_score(y, knn.predict(embedding))
            )
    return results
