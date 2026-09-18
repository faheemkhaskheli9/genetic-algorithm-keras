"""Tests for src/logging_genetic_algorithm.py."""

import logging

from logging_genetic_algorithm import log_generation_stats, logger


def test_logger_is_configured_with_a_console_and_file_handler():
    assert logger.name == "genetic_algorithm"
    assert logger.level == logging.DEBUG
    handler_types = [type(h).__name__ for h in logger.handlers]
    assert "StreamHandler" in handler_types
    # FileHandler is best-effort (skipped if the filesystem refuses it),
    # but in a normal test run it should be present.
    assert "FileHandler" in handler_types


def test_log_generation_stats_computes_best_avg_worst(caplog):
    with caplog.at_level(logging.INFO, logger="genetic_algorithm"):
        stats = log_generation_stats(2, [0.5, 0.9, 0.1])

    assert stats == {"generation": 2, "best": 0.9, "average": 0.5, "worst": 0.1}
    assert any("Generation 2 stats" in record.message for record in caplog.records)


def test_log_generation_stats_handles_empty_scores(caplog):
    with caplog.at_level(logging.WARNING, logger="genetic_algorithm"):
        stats = log_generation_stats(0, [])

    assert stats is None
    assert any("no fitness scores" in record.message for record in caplog.records)


def test_log_generation_stats_writes_to_the_log_file(tmp_path, monkeypatch):
    import logging_genetic_algorithm as lga

    log_file = tmp_path / "generation_stats.log"
    file_logger = logging.getLogger("genetic_algorithm.test_file_write")
    file_logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    file_logger.addHandler(handler)
    file_logger.propagate = False

    try:
        lga.log_generation_stats(1, [1.0, 2.0, 3.0], log=file_logger)
    finally:
        handler.close()
        file_logger.removeHandler(handler)

    contents = log_file.read_text(encoding="utf-8")
    assert "Generation 1 stats" in contents
    assert "best: 3.0000" in contents
