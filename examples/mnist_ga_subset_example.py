"""Runnable example: evolve a tiny Keras CNN on a small MNIST subset.

Usage (from the repo root, with the project's venv active)::

    python examples/mnist_ga_subset_example.py

Trains a small population of randomly-generated Keras models (see
``src/individual_model.py`` / ``src/genetic_algorithm.py``) for a couple of
generations on a two-digit, few-dozen-image MNIST subset, downsampled to
7x7 so a full run finishes on CPU in well under a minute. At the end it
prints the best genome (``GeneticModel.Genes``) found and its fitness.

Network note -- this is the one boundary in the repo that touches the
network: ``load_mnist_binary_subset`` downloads MNIST via
``keras.datasets.mnist.load_data()`` on first use (cached by Keras under
``~/.keras/datasets`` afterwards; no paid API, just a one-time public
dataset fetch). If that download is unavailable -- no network, a
sandboxed/offline session, a flaky CI runner -- it falls back to a small
deterministic synthetic dataset with the same shape, so the example still
runs end-to-end. ``tests/test_mnist_ga_subset_example.py`` always exercises
that fallback path explicitly (via an injected failing loader) so the test
suite itself never depends on network access.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from genetic_algorithm import GeneticAlgorithm  # noqa: E402
from keras import Input  # noqa: E402
from keras.layers import Dense, Flatten  # noqa: E402
from logging_genetic_algorithm import logger  # noqa: E402

MnistLoader = Callable[[], tuple]


def _synthetic_subset(n_train: int, n_test: int, image_size: tuple[int, int]):
    """Deterministic, network-free stand-in with the same shape as the real
    MNIST subset. Used whenever the real download isn't available."""

    h, w = image_size
    rng = np.random.default_rng(0)
    X_train = rng.random((n_train, h, w, 1)).astype("float32")
    y_train = rng.integers(0, 2, size=(n_train, 1)).astype("float32")
    X_test = rng.random((n_test, h, w, 1)).astype("float32")
    y_test = rng.integers(0, 2, size=(n_test, 1)).astype("float32")
    return X_train, y_train, X_test, y_test


def _block_mean_pool(images: np.ndarray, block: int) -> np.ndarray:
    """Downsample (n, h, w) images to (n, h//block, w//block) via block
    averaging. h and w must be evenly divisible by ``block``."""

    n, h, w = images.shape
    if h % block != 0 or w % block != 0:
        raise ValueError(
            f"image size {(h, w)} is not evenly divisible by block={block}"
        )
    return images.reshape(n, h // block, block, w // block, block).mean(axis=(2, 4))


def _select_and_preprocess(
    x: np.ndarray,
    y: np.ndarray,
    classes: tuple[int, int],
    count: int,
    block: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter ``x``/``y`` down to the two requested digit classes, take up
    to ``count`` samples, downsample, normalize, and binarize the label.

    Returns possibly-empty arrays (0 rows) if the requested classes/count
    don't match anything -- callers must check for that themselves rather
    than assume a non-empty result.
    """

    mask = np.isin(y, classes)
    x, y = x[mask], y[mask]
    x, y = x[:count], y[:count]
    if len(x) == 0:
        return np.empty((0, 0, 0, 1), dtype="float32"), np.empty((0, 1), dtype="float32")

    x = _block_mean_pool(x.astype("float32") / 255.0, block)
    x = np.expand_dims(x, -1)
    y_bin = (y == classes[1]).astype("float32").reshape(-1, 1)
    return x, y_bin


def load_mnist_binary_subset(
    n_train: int = 40,
    n_test: int = 20,
    classes: tuple[int, int] = (0, 1),
    image_size: tuple[int, int] = (7, 7),
    mnist_loader: MnistLoader | None = None,
):
    """Load a tiny two-class MNIST subset, downsampled for CPU speed.

    :param mnist_loader: callable returning ``((x_train, y_train),
        (x_test, y_test))`` exactly like ``keras.datasets.mnist.load_data``.
        Defaults to the real MNIST download. Injectable so callers/tests
        can supply a loader that raises (or a fake dataset) and never
        touch the network.
    :param image_size: must evenly divide 28 (MNIST's native size) by the
        same factor in both dimensions, e.g. (7, 7) (block=4) or (14, 14)
        (block=2).

    Falls back to a small synthetic dataset (same shapes) if the loader
    raises, or if the requested classes/count select zero samples -- this
    keeps the example runnable offline; the fallback is logged, not
    silent.
    """

    if mnist_loader is None:
        from keras.datasets import mnist

        mnist_loader = mnist.load_data

    if 28 % image_size[0] != 0 or 28 % image_size[1] != 0:
        raise ValueError("image_size must evenly divide MNIST's native 28x28")
    block = 28 // image_size[0]

    try:
        (x_train, y_train), (x_test, y_test) = mnist_loader()
    except Exception as exc:  # network unavailable, corrupted cache, etc.
        logger.warning(
            "MNIST download unavailable (%s); using synthetic fallback data", exc
        )
        return _synthetic_subset(n_train, n_test, image_size)

    X_train, y_train_bin = _select_and_preprocess(x_train, y_train, classes, n_train, block)
    X_test, y_test_bin = _select_and_preprocess(x_test, y_test, classes, n_test, block)

    if len(X_train) == 0 or len(X_test) == 0:
        logger.warning(
            "MNIST subset selection for classes=%s produced no samples; "
            "using synthetic fallback data",
            classes,
        )
        return _synthetic_subset(n_train, n_test, image_size)

    return X_train, y_train_bin, X_test, y_test_bin


def run_example(
    n_train: int = 40,
    n_test: int = 20,
    classes: tuple[int, int] = (0, 1),
    image_size: tuple[int, int] = (7, 7),
    generations: int = 2,
    population: int = 4,
    training_epochs: int = 1,
    mutation_rate: float = 10.0,
    verbose: int = 1,
    mnist_loader: MnistLoader | None = None,
):
    """Run the genetic algorithm end-to-end and return ``(ga, best_genes,
    best_fitness)``. Factored out of ``main()`` so tests can call it with
    tiny parameters and an injected (non-network) loader.

    :param mutation_rate: a percentage (0-100), per ``GeneticAlgorithm``'s
        own convention (``np.random.random() * 100 < mutation_rate``) --
        10.0 means roughly a 10% chance per gene per generation, higher
        than the library default of 0.1 so a couple of generations is
        enough to visibly show mutation/breeding in this small demo.
    """

    X_train, y_train, X_test, y_test = load_mnist_binary_subset(
        n_train=n_train,
        n_test=n_test,
        classes=classes,
        image_size=image_size,
        mnist_loader=mnist_loader,
    )

    ga = GeneticAlgorithm(
        X_train,
        y_train,
        X_test,
        y_test,
        first_layer=Input(shape=(image_size[0], image_size[1], 1)),
        last_layer=[Flatten(), Dense(1, activation="sigmoid")],
        generations=generations,
        population=population,
        training_epochs=training_epochs,
        mutation_rate=mutation_rate,
        verbose=verbose,
    )
    ga.evolve()

    best_model, best_fitness = max(ga.best_models, key=lambda entry: entry[1])
    best_genes = best_model.Genes if best_model != 0 else {}
    return ga, best_genes, float(best_fitness)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--training-epochs", type=int, default=1)
    parser.add_argument("--n-train", type=int, default=40)
    parser.add_argument("--n-test", type=int, default=20)
    args = parser.parse_args()

    _, best_genes, best_fitness = run_example(
        n_train=args.n_train,
        n_test=args.n_test,
        generations=args.generations,
        population=args.population,
        training_epochs=args.training_epochs,
    )

    print("\n=== Best genome found ===")
    if best_genes:
        print(best_genes)
        print(f"Fitness: {best_fitness:.4f}")
    else:
        print("No individual outperformed the initial placeholder score.")


if __name__ == "__main__":
    main()
