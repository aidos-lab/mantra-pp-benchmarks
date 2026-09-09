"""Quality measures of the learned embedding.

The Pachner objectives shape the embedding rather than the classifier
head, so accuracy alone does not say whether they did what they were
meant to do. The measures here ask that directly: does embedding
distance track the number of Pachner moves between two triangulations,
does it separate homeomorphism types, and is the type linearly
decodable from the embedding.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

MAX_PAIRS = 20000


@torch.no_grad()
def collect_embeddings(model, loader, cfg):
    """Return the embeddings, labels and walk metadata of a loader.

    Parameters
    ----------
    model : nn.Module
        Model with a ``forward_with_embedding`` method.
    loader : DataLoader
        Loader over the split to embed.
    cfg : RootConfig
        Composed config; only the device is read.

    Returns
    -------
    tuple of ndarray
        ``(embedding, y, walk_base, walk_step)``, the last two being
        None for a dataset without Pachner walks.
    """
    model.eval()
    embeddings, labels, bases, steps = [], [], [], []

    for batch in loader:
        batch = batch.to(cfg.trainer.device)
        _, embedding = model.forward_with_embedding(batch)
        embeddings.append(embedding.detach().cpu())
        labels.append(batch.y.detach().cpu().reshape(-1))
        if "walk_base" in batch and "walk_step" in batch:
            bases.append(batch.walk_base.detach().cpu().reshape(-1))
            steps.append(batch.walk_step.detach().cpu().reshape(-1))

    embedding = torch.cat(embeddings).numpy()
    y = torch.cat(labels).numpy()
    if len(bases) == len(embeddings) and bases:
        return (
            embedding,
            y,
            torch.cat(bases).numpy(),
            torch.cat(steps).numpy(),
        )
    return embedding, y, None, None


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
    first, second = _subsample_pairs(first, second, rng)

    distance = np.linalg.norm(embedding[first] - embedding[second], axis=-1)
    moves = np.abs(walk_step[first] - walk_step[second]) * moves_per_step
    if len(np.unique(moves)) < 2 or len(np.unique(distance)) < 2:
        return None
    return float(spearmanr(distance, moves).statistic)


def _subsample_pairs(
    first: np.ndarray, second: np.ndarray, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Cap a pair list at ``MAX_PAIRS`` entries."""
    if len(first) <= MAX_PAIRS:
        return first, second
    keep = rng.choice(len(first), size=MAX_PAIRS, replace=False)
    return first[keep], second[keep]


def _class_separation_auroc(
    embedding: np.ndarray, y: np.ndarray, rng: np.random.Generator
) -> Optional[float]:
    """AUROC of "different class" scored by embedding distance."""
    if len(np.unique(y)) < 2 or len(y) < 2:
        return None
    n_pairs = min(MAX_PAIRS, len(y) * (len(y) - 1) // 2)
    first = rng.integers(0, len(y), size=n_pairs)
    second = rng.integers(0, len(y), size=n_pairs)
    keep = first != second
    first, second = first[keep], second[keep]

    different = (y[first] != y[second]).astype(int)
    if len(np.unique(different)) < 2:
        return None
    distance = np.linalg.norm(embedding[first] - embedding[second], axis=-1)
    return float(roc_auc_score(different, distance))


def _probe_accuracies(
    train_embedding: np.ndarray,
    train_y: np.ndarray,
    embedding: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    """Balanced accuracy of a 5-NN and a linear probe on the embedding."""
    if len(np.unique(train_y)) < 2:
        return {}

    n_neighbors = min(5, len(train_y))
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(train_embedding, train_y)
    probe = LogisticRegression(max_iter=1000)
    probe.fit(train_embedding, train_y)

    return {
        "emb_knn_bal_acc": float(
            balanced_accuracy_score(y, knn.predict(embedding))
        ),
        "emb_probe_bal_acc": float(
            balanced_accuracy_score(y, probe.predict(embedding))
        ),
    }


def evaluate_embeddings(
    model,
    train_loader,
    eval_loader,
    cfg,
    prefix: str = "test",
) -> Dict[str, float]:
    """Measure how well the embedding encodes walks and classes.

    Parameters
    ----------
    model : nn.Module
        Trained model with a ``forward_with_embedding`` method.
    train_loader : DataLoader
        Loader whose embeddings serve as the index of the k-NN
        classifier and the training set of the linear probe.
    eval_loader : DataLoader
        Loader of the split to evaluate.
    cfg : RootConfig
        Composed config; the device and ``dataset.moves_per_step`` are
        read.
    prefix : str
        Metric prefix, e.g. ``"test"`` or ``"ood"``.

    Returns
    -------
    dict of str to float
        ``<prefix>/emb_spearman`` (rank correlation of embedding
        distance with the number of moves between two snapshots of one
        walk), ``<prefix>/emb_auroc`` (separation of the homeomorphism
        types by distance), ``<prefix>/emb_knn_bal_acc`` and
        ``<prefix>/emb_probe_bal_acc``. Measures that the split cannot
        support (no walks, a single class) are left out.
    """
    rng = np.random.default_rng(cfg.seed)
    train_embedding, train_y, _, _ = collect_embeddings(
        model, train_loader, cfg
    )
    embedding, y, walk_base, walk_step = collect_embeddings(
        model, eval_loader, cfg
    )

    metrics = {}
    spearman = _distance_walk_correlation(
        embedding,
        walk_base,
        walk_step,
        cfg.dataset.moves_per_step,
        rng,
    )
    if spearman is not None:
        metrics["emb_spearman"] = spearman

    auroc = _class_separation_auroc(embedding, y, rng)
    if auroc is not None:
        metrics["emb_auroc"] = auroc

    metrics.update(_probe_accuracies(train_embedding, train_y, embedding, y))

    return {f"{prefix}/{name}": value for name, value in metrics.items()}
