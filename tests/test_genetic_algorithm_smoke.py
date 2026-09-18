"""End-to-end smoke test: the genetic algorithm actually runs using the
new individual_model / logging_genetic_algorithm modules.

Uses a tiny synthetic dataset, a tiny population, one generation and one
training epoch, so it stays fast and CPU-only -- no dataset download, no
GPU, no external service.
"""

import numpy as np
from keras import Input
from keras.layers import Dense, Flatten

from genetic_algorithm import GeneticAlgorithm


def _tiny_binary_dataset():
    rng = np.random.default_rng(1)
    X_train = rng.random((8, 6, 6, 1)).astype("float32")
    y_train = rng.integers(0, 2, size=(8, 1)).astype("float32")
    X_test = rng.random((4, 6, 6, 1)).astype("float32")
    y_test = rng.integers(0, 2, size=(4, 1)).astype("float32")
    return X_train, y_train, X_test, y_test


def test_evolve_runs_end_to_end_and_writes_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    X_train, y_train, X_test, y_test = _tiny_binary_dataset()

    ga = GeneticAlgorithm(
        X_train,
        y_train,
        X_test,
        y_test,
        first_layer=Input(shape=(6, 6, 1)),
        last_layer=[Flatten(), Dense(1, activation="sigmoid")],
        generations=1,
        population=3,
        training_epochs=1,
        mutation_rate=0.1,
        verbose=0,
    )

    ga.evolve()

    assert len(ga.statistics) == 1
    assert len(ga.statistics[0]["accuracy"]) == 3
    assert (tmp_path / "results" / "best_models.txt").exists()
    assert (tmp_path / "images" / "Generation 0.png").exists()
