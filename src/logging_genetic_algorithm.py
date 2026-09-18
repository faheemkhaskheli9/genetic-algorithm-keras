"""Logging for the genetic algorithm's run.

Exposes:

- ``logger``: a standard ``logging.Logger`` that writes to both the
  console (INFO and above) and a log file (DEBUG and above), used by
  ``genetic_algorithm.py`` for per-model/per-generation debug messages.
- ``log_generation_stats``: logs the best/average/worst fitness of a
  generation's population, to both handlers via ``logger``.
"""

from __future__ import annotations

import logging
import statistics
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_DIR / "genetic_algorithm.log"

_LOGGER_NAME = "genetic_algorithm"


def _build_logger() -> logging.Logger:
    log = logging.getLogger(_LOGGER_NAME)
    log.setLevel(logging.DEBUG)

    if not log.handlers:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        log.addHandler(console_handler)

        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            log.addHandler(file_handler)
        except OSError:
            # A read-only filesystem or missing permissions shouldn't stop
            # the algorithm from running -- console logging still works.
            log.warning("Could not open log file %s; logging to console only.", LOG_FILE)

        log.propagate = False

    return log


logger = _build_logger()


def log_generation_stats(generation: int, fitness_scores, log: logging.Logger = logger):
    """Log best/average/worst fitness for one generation.

    :param generation: generation index (0-based).
    :param fitness_scores: iterable of per-model fitness values for this
        generation's population.
    :param log: logger to write to (defaults to this module's ``logger``).
    :return: dict with the computed stats, or ``None`` if there were no
        scores to report.
    """

    fitness_scores = list(fitness_scores)
    if not fitness_scores:
        log.warning("Generation %s has no fitness scores to report.", generation)
        return None

    best = max(fitness_scores)
    worst = min(fitness_scores)
    average = statistics.fmean(fitness_scores)

    log.info(
        "Generation %s stats -> best: %.4f, avg: %.4f, worst: %.4f",
        generation,
        best,
        average,
        worst,
    )

    return {
        "generation": generation,
        "best": best,
        "average": average,
        "worst": worst,
    }
