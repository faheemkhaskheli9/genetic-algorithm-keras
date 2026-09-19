"""Tests for the runnable example (examples/mnist_ga_subset_example.py).

The real MNIST download (``keras.datasets.mnist.load_data``) is the one
network boundary in this repo. Per the "CPU-only, no paid APIs or hardware"
project constraint, this test suite never depends on that download actually
succeeding: every test either injects a ``mnist_loader`` that raises (which
forces the documented synthetic fallback) or drives the real-data
preprocessing helper directly with an in-memory fake array shaped like
MNIST -- so the real-path logic (class filtering, downsampling,
normalization, label binarization) is exercised without ever touching the
network.
"""

import sys
from pathlib import Path

import numpy as np

_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from mnist_ga_subset_example import (  # noqa: E402
    _block_mean_pool,
    _select_and_preprocess,
    load_mnist_binary_subset,
    run_example,
)


def _raising_loader():
    raise RuntimeError("no network available in this test")


def _fake_mnist_loader():
    """A tiny in-memory stand-in for keras.datasets.mnist.load_data():
    20 train + 10 test 28x28 uint8 images, labels cycling 0-9, so both
    train and test contain the default (0, 1) classes."""

    rng = np.random.default_rng(3)

    def _make(n):
        x = rng.integers(0, 256, size=(n, 28, 28), dtype=np.uint8)
        y = np.arange(n) % 10
        return x, y

    return _make(20), _make(10)


def test_block_mean_pool_downsamples_and_averages():
    images = np.zeros((1, 4, 4), dtype="float32")
    images[0, :2, :2] = 1.0  # top-left 2x2 block is all 1s, rest 0

    pooled = _block_mean_pool(images, block=2)

    assert pooled.shape == (1, 2, 2)
    assert pooled[0, 0, 0] == 1.0
    assert pooled[0, 0, 1] == 0.0
    assert pooled[0, 1, 0] == 0.0


def test_block_mean_pool_rejects_non_divisible_size():
    images = np.zeros((1, 5, 5), dtype="float32")
    try:
        _block_mean_pool(images, block=2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a non-divisible block size")


def test_select_and_preprocess_filters_classes_and_binarizes_labels():
    rng = np.random.default_rng(4)
    x = rng.integers(0, 256, size=(12, 28, 28), dtype=np.uint8)
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])

    X, y_bin = _select_and_preprocess(x, y, classes=(0, 1), count=100, block=4)

    # only the 8 samples labeled 0 or 1 survive the class filter
    assert X.shape == (8, 7, 7, 1)
    assert X.min() >= 0.0 and X.max() <= 1.0
    assert set(np.unique(y_bin)).issubset({0.0, 1.0})
    # label 1 maps to 1.0, label 0 maps to 0.0 (classes[1] is the positive class)
    assert y_bin.sum() == 4


def test_select_and_preprocess_returns_empty_when_classes_absent():
    x = np.zeros((5, 28, 28), dtype=np.uint8)
    y = np.array([2, 3, 4, 5, 6])  # none are in classes=(0, 1)

    X, y_bin = _select_and_preprocess(x, y, classes=(0, 1), count=100, block=4)

    assert len(X) == 0
    assert len(y_bin) == 0


def test_load_mnist_binary_subset_uses_real_loader_when_it_succeeds():
    X_train, y_train, X_test, y_test = load_mnist_binary_subset(
        n_train=100, n_test=100, image_size=(7, 7), mnist_loader=_fake_mnist_loader
    )

    assert X_train.shape[1:] == (7, 7, 1)
    assert X_test.shape[1:] == (7, 7, 1)
    assert len(X_train) > 0 and len(X_test) > 0


def test_load_mnist_binary_subset_falls_back_to_synthetic_data_without_network():
    X_train, y_train, X_test, y_test = load_mnist_binary_subset(
        n_train=6, n_test=4, image_size=(7, 7), mnist_loader=_raising_loader
    )

    assert X_train.shape == (6, 7, 7, 1)
    assert X_test.shape == (4, 7, 7, 1)
    assert set(np.unique(y_train)).issubset({0.0, 1.0})


def test_load_mnist_binary_subset_rejects_non_divisible_image_size():
    try:
        load_mnist_binary_subset(image_size=(9, 9), mnist_loader=_raising_loader)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an image_size that doesn't divide 28")


def test_run_example_completes_and_reports_a_best_genome(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    ga, best_genes, best_fitness = run_example(
        n_train=6,
        n_test=4,
        image_size=(7, 7),
        generations=1,
        population=2,
        training_epochs=1,
        verbose=0,
        mnist_loader=_raising_loader,
    )

    assert len(ga.statistics) == 1
    assert isinstance(best_genes, dict)
    assert "layers" in best_genes
    assert isinstance(best_fitness, float)
    assert (tmp_path / "results" / "best_models.txt").exists()
