"""Individual representation for the genetic algorithm.

An "individual" is one candidate Keras model in the population. It is made
of two parts:

- ``GENES``: the default search space (which layer types exist, what
  parameter values/ranges each layer type can randomly take, and how gene
  names map onto real Keras layers). ``genetic_algorithm.py`` reads this to
  create/mutate/breed individuals.
- ``GeneticModel``: one concrete individual -- a dict of genes plus the
  compiled ``keras.Model`` built from those genes.

``GENES`` is derived entirely from ``default_config.DEFAULT_GENES`` so the
module works standalone (no ``myconfig.py`` required). A project can still
supply its own ``myconfig.py`` (see ``set_config.py`` / README) to override
the search space; if present it is merged over these defaults.
"""

from __future__ import annotations

from typing import Any

import keras
from keras import layers as keras_layers

from default_config import DEFAULT_GENES


def _build_default_genes() -> dict:
    """Build a self-contained GENES search space from DEFAULT_GENES.

    Convolution strides are intentionally kept fixed at 1 (not randomized)
    and padding fixed at 'same': this guarantees every layer preserves the
    spatial shape of its input, which is what makes the residual block's
    concatenation of the block's input with its output always shape-safe.
    """

    genes: dict[str, Any] = {
        "layers": [1, 2, 3],
        "layer_choice": ["conv2d", "batch_norm", "activation", "res_block"],
        "keras_mapping": {
            "conv2d": "Conv2D",
            "batch_norm": "BatchNormalization",
            "activation": "Activation",
            "res_block": "res_block",
        },
        "conv2d": {
            "filters": [8, 16, 32],
            "kernel_size": [3, 5],
            "strides": 1,
            "activation": "relu",
            "padding": "same",
            "dilation_rate": 1,
        },
        "batch_norm": {
            "momentum": DEFAULT_GENES["BatchNormalization"]["momentum"],
        },
        "activation": {
            "function": list(DEFAULT_GENES["Activation"]["function"]),
        },
        "res_block": {
            "layers": DEFAULT_GENES["res_block"]["layers"],
            "layer_choice": ["conv2d"],
        },
    }
    return genes


def _merge_user_config(genes: dict) -> dict:
    """Merge an optional user-supplied ``myconfig.my_genes`` over the defaults.

    ``myconfig.py`` is a project-local, git-ignored file (see README) that
    is not part of this package. If it doesn't exist we simply keep the
    defaults -- that's the expected case for a fresh checkout.
    """

    try:
        from myconfig import my_genes  # type: ignore
    except ImportError:
        return genes

    merged = dict(genes)
    for key, value in my_genes.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


GENES = _merge_user_config(_build_default_genes())


def _apply_layer_chain(x, layer_or_layers):
    """Apply a single Keras layer, or a sequence of layers, to tensor x."""

    if isinstance(layer_or_layers, (list, tuple)):
        for layer in layer_or_layers:
            x = layer(x)
        return x
    return layer_or_layers(x)


def _clone_layer(layer):
    """Return a fresh, unbuilt copy of a Keras layer (same config, no weights)."""

    return layer.__class__.from_config(layer.get_config())


def _clone_layer_chain(layer_or_layers):
    if isinstance(layer_or_layers, (list, tuple)):
        return [_clone_layer(layer) for layer in layer_or_layers]
    return _clone_layer(layer_or_layers)


class GeneticModel:
    """One individual in the genetic algorithm's population.

    :param genes: dict describing this individual's architecture. Must
        contain a ``layers_config`` list of layer-gene dicts, each with a
        ``name`` key of ``'Conv2D'``, ``'BatchNormalization'``,
        ``'Activation'`` or ``'res_block'``.
    :param training_epochs: epochs to train for in :meth:`evaluate_model`.
    :param first_layer: a Keras input tensor, e.g. ``keras.Input(shape=...)``.
    :param last_layer: a Keras layer, or list of layers applied in sequence
        (e.g. ``[Flatten(), Dense(1, activation='sigmoid')]``), producing
        the model's output tensor from the last hidden layer's output.
    :param verbose: verbosity level (see README "Verbose Level" table).
    :param loss: Keras loss name/instance used to compile the model.
    :param metrics: list of Keras metric names used to compile the model.
    """

    def __init__(
        self,
        genes: dict,
        training_epochs: int = 10,
        first_layer=None,
        last_layer=None,
        verbose: int = 0,
        loss: str = "binary_crossentropy",
        metrics: list | None = None,
    ) -> None:
        if first_layer is None or last_layer is None:
            raise ValueError(
                "GeneticModel requires both first_layer and last_layer "
                "(e.g. keras.Input(shape=...) and a Dense output layer)."
            )

        self.Genes = genes
        self.training_epochs = training_epochs
        self.first_layer = first_layer
        self.last_layer = last_layer
        self.verbose = verbose
        self.loss = loss
        self.metrics = metrics if metrics is not None else ["accuracy"]

        self.Model = self.build_model()

    def _build_layer(self, layer_conf: dict):
        """Turn one layer-gene dict into a fresh (uncalled) Keras layer."""

        name = layer_conf["name"]
        if name == "Conv2D":
            return keras_layers.Conv2D(
                filters=layer_conf["filters"],
                kernel_size=layer_conf["kernel_size"],
                strides=layer_conf["strides"],
                activation=layer_conf["activation"],
                padding=layer_conf["padding"],
                dilation_rate=layer_conf["dilation_rate"],
            )
        if name == "BatchNormalization":
            return keras_layers.BatchNormalization(momentum=layer_conf["momentum"])
        if name == "Activation":
            return keras_layers.Activation(layer_conf["function"])
        raise ValueError(f"Unknown layer gene name: {name!r}")

    def _apply_res_block(self, x, layer_conf: dict):
        """Apply a residual block: n inner layers, concatenated with the
        block's own input, per the README's residual-block description."""

        block_input = x
        for inner_conf in layer_conf["layers_config"]:
            x = self._build_layer(inner_conf)(x)
        return keras_layers.Concatenate()([block_input, x])

    def build_model(self) -> keras.Model:
        """Build (but do not train) the keras.Model described by self.Genes.

        ``self.last_layer`` is cloned (fresh weights, same config) before
        use: it is the same object shared by every individual in the
        population (passed once by ``GeneticAlgorithm``), and applying it
        directly would silently tie every individual's output-layer
        weights together instead of letting each evolve independently.
        ``self.first_layer`` is a weightless ``keras.Input`` tensor, so
        reusing it as-is across individuals is safe.
        """

        x = self.first_layer
        for layer_conf in self.Genes.get("layers_config", []):
            if layer_conf["name"] == "res_block":
                x = self._apply_res_block(x, layer_conf)
            else:
                x = self._build_layer(layer_conf)(x)

        outputs = _apply_layer_chain(x, _clone_layer_chain(self.last_layer))
        model = keras.Model(inputs=self.first_layer, outputs=outputs)
        model.compile(optimizer="adam", loss=self.loss, metrics=self.metrics)
        return model

    def evaluate_model(self, X_train, y_train, X_test, y_test):
        """Train for ``self.training_epochs`` and return a fitness score.

        Higher is always better: if the compiled model reports an
        'accuracy' metric that's returned directly, otherwise the fitness
        is the negative test loss (so lower loss -> higher fitness).
        """

        self.Model.fit(
            X_train,
            y_train,
            epochs=self.training_epochs,
            verbose=0,
        )
        results = self.Model.evaluate(X_test, y_test, verbose=0, return_dict=True)

        for key in ("accuracy", "compile_metrics"):
            if key in results:
                return float(results[key])
        return -float(results["loss"])
