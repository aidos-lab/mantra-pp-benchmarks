"""Tests for the embedding-quality measures."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from src.metrics.embedding import collect_embeddings, evaluate_embeddings


class IdentityModel(torch.nn.Module):
    """Model whose embedding is the input feature of each entry."""

    def forward_with_embedding(self, batch):
        return batch.x, batch.x

    def eval(self):
        return self


def make_loader(features, labels, walk_base=None, walk_step=None):
    """One batch holding the given per-entry features."""
    entries = []
    for index, (feature, label) in enumerate(zip(features, labels)):
        data = Data(x=torch.tensor(feature).float().reshape(1, -1))
        data.y = torch.tensor(int(label))
        if walk_base is not None:
            data.walk_base = int(walk_base[index])
            data.walk_step = int(walk_step[index])
        entries.append(data)
    return [Batch.from_data_list(entries)]


def make_cfg(moves_per_step=1):
    return SimpleNamespace(
        seed=0,
        trainer=SimpleNamespace(device="cpu"),
        dataset=SimpleNamespace(moves_per_step=moves_per_step),
    )


class TestCollectEmbeddings:
    def test_walk_metadata_is_returned_when_present(self):
        loader = make_loader(
            [[0.0], [1.0]], [0, 0], walk_base=[0, 0], walk_step=[0, 1]
        )

        embedding, y, base, step = collect_embeddings(
            IdentityModel(), loader, make_cfg()
        )

        assert embedding.shape == (2, 1)
        assert y.tolist() == [0, 0]
        assert base.tolist() == [0, 0]
        assert step.tolist() == [0, 1]

    def test_missing_walk_metadata_yields_none(self):
        loader = make_loader([[0.0], [1.0]], [0, 1])

        _, _, base, step = collect_embeddings(
            IdentityModel(), loader, make_cfg()
        )

        assert base is None and step is None


class TestEvaluateEmbeddings:
    def test_distance_proportional_to_moves_gives_perfect_correlation(self):
        # One walk whose snapshots sit on a line at their move distance.
        features = [[0.0], [3.0], [6.0], [9.0]]
        loader = make_loader(
            features, [0] * 4, walk_base=[0] * 4, walk_step=[0, 1, 2, 3]
        )

        metrics = evaluate_embeddings(
            IdentityModel(), loader, loader, make_cfg(moves_per_step=3)
        )

        assert metrics["test/emb_spearman"] == pytest.approx(1.0)

    def test_separated_classes_are_perfectly_decodable(self):
        rng = np.random.default_rng(0)
        features = np.concatenate(
            [rng.normal(0, 0.01, (20, 2)), rng.normal(10, 0.01, (20, 2))]
        )
        labels = [0] * 20 + [1] * 20
        loader = make_loader(features.tolist(), labels)

        metrics = evaluate_embeddings(
            IdentityModel(), loader, loader, make_cfg(), prefix="ood"
        )

        assert metrics["ood/emb_auroc"] == pytest.approx(1.0, abs=0.01)
        assert metrics["ood/emb_knn_bal_acc"] == pytest.approx(1.0)
        assert metrics["ood/emb_probe_bal_acc"] == pytest.approx(1.0)
        assert "ood/emb_spearman" not in metrics

    def test_uninformative_embedding_scores_at_chance(self):
        rng = np.random.default_rng(1)
        features = rng.normal(0, 1, (40, 2))
        labels = [0, 1] * 20
        loader = make_loader(features.tolist(), labels)

        metrics = evaluate_embeddings(
            IdentityModel(), loader, loader, make_cfg()
        )

        assert metrics["test/emb_auroc"] == pytest.approx(0.5, abs=0.15)
        assert metrics["test/emb_probe_bal_acc"] == pytest.approx(0.5, abs=0.2)
