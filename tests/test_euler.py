import torch

from mantra.transforms import NAME_TO_CLASS_2M

from src.metrics.euler import chi_balanced_accuracy, chi_of_classes, chi_of_surface


def test_chi_of_the_nine_classes():
    expected = {
        "S^2": 2, "RP^2": 1, "T^2": 0, "Klein bottle": 0, "#^3 RP^2": -1,
        "#^2 T^2": -2, "#^4 RP^2": -2, "#^5 RP^2": -3, "#^6 RP^2": -4,
    }
    assert {name: chi_of_surface(name) for name in expected} == expected


def test_chi_of_classes_follows_the_contiguous_indices():
    label_to_index = {NAME_TO_CLASS_2M["T^2"]: 0, NAME_TO_CLASS_2M["Klein bottle"]: 1, NAME_TO_CLASS_2M["S^2"]: 2}
    assert chi_of_classes(label_to_index).tolist() == [0, 0, 2]


def test_chi_only_classifier_is_perfect_at_chi_level():
    chi = torch.tensor([0, 0, 2])  # torus, Klein bottle, sphere
    y = torch.tensor([0, 1, 1, 2, 2])
    y_hat = torch.tensor([1, 0, 0, 2, 2])  # torus <-> Klein bottle confused, sphere right
    assert chi_balanced_accuracy(y, y_hat, chi) == 1.0
    assert chi_balanced_accuracy(y, torch.tensor([2, 2, 2, 0, 0]), chi) == 0.0
