import sys
from contextlib import contextmanager
from io import StringIO
import pandas as pd


@contextmanager
def capture(command, *args, **kwargs):
    """Context manager to capture stdout from a command."""
    out, sys.stdout = sys.stdout, StringIO()
    command(*args, **kwargs)
    sys.stdout.seek(0)
    yield sys.stdout.read()
    sys.stdout = out


# Shared test data fixtures to reduce duplication across test files
def get_basic_test_data():
    """Get basic test data with common names for testing."""
    return pd.DataFrame(
        [
            {"last": "smith", "first": "john"},
            {"last": "garcia", "first": "maria"},
            {"last": "zhang", "first": "wei"},
        ]
    )


def get_asian_hispanic_test_data():
    """Get test data focused on Asian and Hispanic names."""
    return pd.DataFrame(
        [
            {"last": "zhang", "first": "simon", "true_race": "asian"},
            {"last": "torres", "first": "raul", "true_race": "hispanic"},
        ]
    )


def get_ordered_test_data():
    """Get test data with ID column for order preservation testing."""
    return pd.DataFrame(
        [
            {"last": "smith", "first": "john", "id": 1},
            {"last": "zhang", "first": "wei", "id": 2},
            {"last": "garcia", "first": "maria", "id": 3},
        ]
    )


def get_census_test_data():
    """Get test data for Census-specific testing."""
    return pd.DataFrame(
        [
            {"last": "smith", "true_race": "nh_white"},
            {"last": "zhang", "true_race": "asian"},
        ]
    )


# Valid race/ethnicity categories used across all models
VALID_RACE_CATEGORIES = {"asian", "hispanic", "nh_black", "nh_white", "other"}
