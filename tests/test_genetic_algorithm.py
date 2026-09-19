"""Regression tests for genetic_algorithm.py's random gene generation.

np.random.choice / np.random.randint return numpy scalar types (e.g.
numpy.int64), not native Python int/str. Two bugs followed from that,
both fixed by coercing to native types at the point of selection
(`_to_native`):

1. Keras's own layer argument validation uses `isinstance(value, int)`
   and rejects a bare numpy.int64 -- a hard crash when a randomly-chosen
   `kernel_size` (etc.) reached Conv2D.
2. `mutate_model`'s `type(x) == int` / `type(x) == str` checks use strict
   type equality, which is also False for a numpy scalar -- mutation
   would silently never apply to a randomly-generated gene. It also used
   to call `model[gene]` (GeneticModel has no `__getitem__`), which
   raised whenever mutation triggered at all.
"""

import numpy as np
from keras import Input
from keras.layers import Dense, Flatten

from genetic_algorithm import GeneticAlgorithm
from individual_model import GeneticModel


def _tiny_algorithm(population=2, mutation_rate=0.1):
    rng = np.random.default_rng(2)
    X = rng.random((4, 6, 6, 1)).astype("float32")
    y = rng.integers(0, 2, size=(4, 1)).astype("float32")
    return GeneticAlgorithm(
        X,
        y,
        X,
        y,
        first_layer=Input(shape=(6, 6, 1)),
        last_layer=[Flatten(), Dense(1, activation="sigmoid")],
        generations=1,
        population=population,
        mutation_rate=mutation_rate,
        training_epochs=1,
        verbose=0,
    )


def _assert_no_numpy_scalars(genes):
    for key, value in genes.items():
        if key == "layers_config":
            continue
        assert not isinstance(value, np.generic), f"{key} is a numpy scalar: {value!r}"
    for layer_conf in genes.get("layers_config", []):
        for key, value in layer_conf.items():
            assert not isinstance(value, np.generic), f"{key} is a numpy scalar: {value!r}"


def test_create_model_from_initial_genes_uses_native_python_types():
    ga = _tiny_algorithm(population=1)
    model = ga.models[0]
    assert type(model.Genes["layers"]) is int
    _assert_no_numpy_scalars(model.Genes)


def test_mutate_model_always_triggering_does_not_crash():
    ga = _tiny_algorithm(population=1, mutation_rate=100)
    model = ga.models[0]

    mutated = ga.mutate_model(model)

    assert type(mutated.Genes["layers"]) is int
    _assert_no_numpy_scalars(mutated.Genes)


def test_breeding_produces_native_layer_count():
    ga = _tiny_algorithm(population=2)
    child = ga.breeding(ga.models[0], ga.models[1])
    assert type(child.Genes["layers"]) is int


def test_create_population_builds_one_model_per_requested_individual():
    population_size = 4
    ga = _tiny_algorithm(population=population_size)

    assert len(ga.models) == population_size
    for model in ga.models:
        assert isinstance(model, GeneticModel)
        # Each individual gets its own freshly-built keras.Model, not a
        # shared reference -- would silently make every "individual" the
        # same model and defeat the point of a population.
        assert model.Model is not None
    model_ids = {id(model) for model in ga.models}
    assert len(model_ids) == population_size
