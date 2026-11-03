"""Public logging output and performance tracking contracts."""

import json
import logging
import sys

import pytest

from ethnicolr2.logging_config import (
    EthnicolrLogger,
    PerformanceTracker,
    StructuredFormatter,
    performance_logged,
)


@pytest.fixture
def isolated_logging(monkeypatch):
    root = logging.getLogger("ethnicolr2")
    handlers, level = root.handlers[:], root.level
    monkeypatch.setattr(EthnicolrLogger, "_configured", False)
    monkeypatch.setattr(EthnicolrLogger, "_loggers", {})
    monkeypatch.delenv("ETHNICOLR_LOG_LEVEL", raising=False)
    yield
    for handler in root.handlers:
        if handler not in handlers:
            handler.close()
    root.handlers[:] = handlers
    root.setLevel(level)


def test_formatter_preserves_message_metadata_and_exception():
    try:
        raise ValueError("invalid input")
    except ValueError:
        record = logging.LogRecord(
            "ethnicolr2.test",
            logging.ERROR,
            __file__,
            42,
            "bad %s",
            ("row",),
            sys.exc_info(),
            "parse",
        )
    record.request_id = "req-123"
    fields = json.loads(StructuredFormatter().format(record))
    assert fields["message"] == "bad row"
    assert fields["level"] == "ERROR"
    assert fields["logger"] == "ethnicolr2.test"
    assert fields["function"] == "parse"
    assert fields["module"] == "test_logging_config"
    assert fields["timestamp"]
    assert fields["request_id"] == "req-123"
    assert "ValueError: invalid input" in fields["exception"]
    selected = json.loads(StructuredFormatter(["message", "line"]).format(record))
    assert selected["line"] == 42
    assert "timestamp" not in selected
    assert "logger" not in selected


def test_setup_writes_json_file_once_and_honors_environment(
    isolated_logging, monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("ETHNICOLR_LOG_LEVEL", "debug")
    destination = tmp_path / "nested" / "events.jsonl"
    EthnicolrLogger.setup(log_file=str(destination))
    logger = EthnicolrLogger.get_logger("worker")
    assert logger is EthnicolrLogger.get_logger("worker")
    logger.debug("processed", extra={"rows": 3})
    EthnicolrLogger.setup(log_file=str(destination))
    lines = [json.loads(line) for line in destination.read_text().splitlines()]
    assert [line["message"] for line in lines] == ["Logging configured", "processed"]
    assert lines[-1]["rows"] == 3
    assert lines[-1]["level"] == "DEBUG"
    assert len(capsys.readouterr().out.splitlines()) == 2


def test_simple_console_format(isolated_logging, capsys):
    EthnicolrLogger.setup(format_type="simple")
    EthnicolrLogger.get_logger("worker").warning("check input")
    assert "ethnicolr2.worker - WARNING - check input" in capsys.readouterr().out


def test_logger_initializes_on_demand(isolated_logging, capsys):
    EthnicolrLogger.get_logger("worker").info("ready")
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["message"] for line in lines] == ["Logging configured", "ready"]


def test_performance_reports_duration_and_metadata(monkeypatch, caplog):
    ticks = iter([10.0, 12.5])
    monkeypatch.setattr(
        "ethnicolr2.logging_config.time.perf_counter", lambda: next(ticks)
    )
    logger = logging.getLogger("performance-contract")
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        with PerformanceTracker("load", logger, rows=4):
            pass
    start, complete = caplog.records
    assert start.event == "start"
    assert complete.event == "complete"
    assert complete.operation == "load"
    assert complete.rows == 4
    assert complete.duration_seconds == 2.5


def test_performance_logs_failure_without_swallowing_exception(caplog):
    logger = logging.getLogger("performance-error-contract")
    with pytest.raises(ValueError, match="invalid"):
        with PerformanceTracker("load", logger):
            raise ValueError("invalid")
    record = caplog.records[-1]
    assert record.event == "error"
    assert record.error_type == "ValueError"
    assert record.duration_seconds >= 0


def test_performance_decorator_preserves_result(isolated_logging, capsys):
    @performance_logged("add")
    def add(left, *, right):
        return left + right

    assert add(2, right=3) == 5
    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert output[-1]["event"] == "complete"
    assert output[-1]["operation"] == "add"


def test_tracker_exit_without_entry_does_not_report_duration(caplog):
    PerformanceTracker("unstarted", logging.getLogger("unstarted")).__exit__(
        None, None, None
    )
    assert not caplog.records
