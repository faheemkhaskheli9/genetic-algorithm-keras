# Architecture

## Pipeline

```
GENES (search space)
   |
   v
GeneticAlgorithm.create_population()  -- create_model_from_initial_genes()
   |                                       builds a random Genes dict per GENES
   v
GeneticModel(genes, ...)               -- src/individual_model.py
   |   build_model(): first_layer -> [Conv2D | BatchNormalization |
   |                   Activation | res_block]* -> last_layer -> keras.Model
   v
GeneticAlgorithm.survival_of_fittest() -- trains + evaluates each
   |                                      individual (GeneticModel.evaluate_model)
   |                                      logs generation stats
   v
GeneticAlgorithm.create_new_best_generation() -- breeding() + mutate_model()
   |
   (repeat for `generations`)
```

## Components

- **`src/default_config.py`** -- `DEFAULT_GENES`: the base per-keras-layer
  parameter defaults (Conv2D, Dense, BatchNormalization, Activation,
  res_block). Not meant to be edited; it's the fallback when no
  project-specific config is supplied.
- **`src/set_config.py`** -- optional helper that merges a project-local
  `myconfig.py` (git-ignored, not shipped) over `default_config.DEFAULT_GENES`.
  Not imported by the algorithm itself; see README "myconfig.py" section.
- **`src/individual_model.py`** -- `GENES` (the concrete, self-contained
  search space used by the algorithm, built from `DEFAULT_GENES`) and
  `GeneticModel` (one individual: a `Genes` dict plus the `keras.Model`
  built from it, with `evaluate_model()` to train+score it).
- **`src/logging_genetic_algorithm.py`** -- `logger` (console + file,
  `logs/genetic_algorithm.log`) and `log_generation_stats()` (best/average/
  worst fitness per generation).
- **`src/genetic_algorithm.py`** -- `GeneticAlgorithm`: population
  creation, fitness evaluation (`survival_of_fittest`), selection and
  breeding (`create_new_best_generation`, `breeding`, `mutate_model`), and
  the `evolve()` driver loop.

## Design notes / boundaries mocked in tests

- All tests use tiny synthetic in-memory arrays (a handful of small
  images) and 1 training epoch -- no external dataset download, no GPU, no
  paid API. This is noted here per the "CPU-only, no paid APIs or
  hardware" project constraint.
- Convolution `strides` are fixed at `1` with `padding='same'` in the
  default `GENES`, so every generated layer preserves spatial shape. This
  is what makes the residual block's `Concatenate([block_input, x])`
  always shape-compatible, regardless of which layers were randomly
  chosen.
