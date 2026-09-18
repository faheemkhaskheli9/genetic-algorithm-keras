"""Tests for src/individual_model.py.

These only use tiny in-memory synthetic arrays and a single training
epoch -- no dataset download, no GPU, no external service. That's the
CPU-only boundary for this project; nothing here needs mocking beyond
keeping the data and epoch count small.
"""

import numpy as np
import pytest
from keras import Input
from keras.layers import Dense, Flatten

from individual_model import GENES, GeneticModel


def _tiny_binary_dataset():
    rng = np.random.default_rng(0)
    X_train = rng.random((8, 6, 6, 1)).astype("float32")
    y_train = rng.integers(0, 2, size=(8, 1)).astype("float32")
    X_test = rng.random((4, 6, 6, 1)).astype("float32")
    y_test = rng.integers(0, 2, size=(4, 1)).astype("float32")
    return X_train, y_train, X_test, y_test


def _conv_gene(filters=4):
    return {
        "name": "Conv2D",
        "filters": filters,
        "kernel_size": 3,
        "strides": 1,
        "activation": "relu",
        "padding": "same",
        "dilation_rate": 1,
    }


def _activation_gene():
    return {"name": "Activation", "function": "relu"}


class TestGenesSearchSpace:
    def test_top_level_keys_present(self):
        for key in ("layers", "layer_choice", "keras_mapping"):
            assert key in GENES

    def test_every_layer_choice_is_mapped_and_configured(self):
        for name in GENES["layer_choice"]:
            assert name in GENES["keras_mapping"], f"{name} missing from keras_mapping"
            assert name in GENES, f"{name} has no config entry in GENES"

    def test_conv2d_gene_uses_fixed_stride_and_same_padding(self):
        # Fixed stride=1 / padding='same' is what keeps every generated
        # layer shape-compatible, which res_block's concatenation depends on.
        assert GENES["conv2d"]["strides"] == 1
        assert GENES["conv2d"]["padding"] == "same"

    def test_res_block_only_offers_shape_preserving_inner_layers(self):
        assert GENES["res_block"]["layer_choice"] == ["conv2d"]


class TestGeneticModel:
    def test_requires_first_and_last_layer(self):
        with pytest.raises(ValueError):
            GeneticModel({"layers_config": [_conv_gene()]})

    def test_builds_a_keras_model_with_expected_io_shape(self):
        genes = {"layers_config": [_conv_gene(), _activation_gene()]}
        model = GeneticModel(
            genes,
            training_epochs=1,
            first_layer=Input(shape=(6, 6, 1)),
            last_layer=[Flatten(), Dense(1, activation="sigmoid")],
        )
        assert model.Model.input_shape == (None, 6, 6, 1)
        assert model.Model.output_shape == (None, 1)

    def test_res_block_gene_builds_without_shape_errors(self):
        genes = {
            "layers_config": [
                {
                    "name": "res_block",
                    "layers_config": [_conv_gene(filters=4), _conv_gene(filters=4)],
                },
            ]
        }
        model = GeneticModel(
            genes,
            training_epochs=1,
            first_layer=Input(shape=(6, 6, 3)),
            last_layer=[Flatten(), Dense(1, activation="sigmoid")],
        )
        assert model.Model.output_shape == (None, 1)
        layer_types = [type(layer).__name__ for layer in model.Model.layers]
        assert "Concatenate" in layer_types

    def test_evaluate_model_trains_and_returns_a_finite_fitness(self):
        X_train, y_train, X_test, y_test = _tiny_binary_dataset()
        genes = {"layers_config": [_conv_gene()]}
        model = GeneticModel(
            genes,
            training_epochs=1,
            first_layer=Input(shape=(6, 6, 1)),
            last_layer=[Flatten(), Dense(1, activation="sigmoid")],
        )
        fitness = model.evaluate_model(X_train, y_train, X_test, y_test)
        assert isinstance(fitness, float)
        assert np.isfinite(fitness)

    def test_last_layer_weights_are_not_shared_across_individuals(self):
        # A shared (uncloned) last_layer would tie every individual's
        # output-layer weights together, silently breaking the "independent
        # population" premise of a genetic algorithm.
        shared_last_layer = [Flatten(), Dense(1, activation="sigmoid")]
        first_layer = Input(shape=(6, 6, 1))
        genes = {"layers_config": [_conv_gene()]}

        model_a = GeneticModel(
            genes, training_epochs=1, first_layer=first_layer, last_layer=shared_last_layer
        )
        model_b = GeneticModel(
            genes, training_epochs=1, first_layer=first_layer, last_layer=shared_last_layer
        )

        dense_a = model_a.Model.layers[-1]
        dense_b = model_b.Model.layers[-1]
        assert dense_a is not dense_b
        assert dense_a.get_weights()[0] is not dense_b.get_weights()[0]
