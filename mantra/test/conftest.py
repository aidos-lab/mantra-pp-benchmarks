"""Shared pytest configuration and fixtures."""

import json

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--dataset-path",
        action="store",
        default=None,
        help="Path to manifold triangulations JSON file",
    )


@pytest.fixture
def dataset_path(request):
    path = request.config.getoption("--dataset-path")
    if path is None:
        pytest.skip("No --dataset-path provided")
    return path


@pytest.fixture
def dataset(dataset_path):
    with open(dataset_path) as f:
        return json.load(f)