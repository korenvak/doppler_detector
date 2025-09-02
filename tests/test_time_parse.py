#!/usr/bin/env python3
"""
Tests for the time parsing functionality.
Run with: python -m pytest tests/test_time_parse.py
"""

import sys
import os
from datetime import datetime

# Add parent directory to path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from spectrogram_gui.utils.time_parse import parse_times_from_filename


def test_dd_mm_variant():
    """Test parsing DD-MM format (day > 12)."""
    name = "pixel - 211 - 2025-19-08 10-19-59 - 2025-19-08 10-28-55"
    px, s, e = parse_times_from_filename(name)
    assert px == 211
    assert (s.year, s.month, s.day, s.hour, s.minute, s.second) == (2025, 8, 19, 10, 19, 59)
    assert (e.year, e.month, e.day, e.hour, e.minute, e.second) == (2025, 8, 19, 10, 28, 55)
    assert (e - s).total_seconds() == 8*60 + 56


def test_iso_variant():
    """Test parsing ISO format (MM-DD where month <= 12)."""
    name = "pixel - 2221 - 2025-05-29 10-46-00 - 2025-05-29 11-08-00"
    px, s, e = parse_times_from_filename(name)
    assert px == 2221
    assert (s.year, s.month, s.day) == (2025, 5, 29)
    assert (s.hour, s.minute, s.second) == (10, 46, 0)
    assert (e.year, e.month, e.day) == (2025, 5, 29)
    assert (e.hour, e.minute, e.second) == (11, 8, 0)
    assert (e - s).total_seconds() == 22*60


def test_ambiguous_date():
    """Test ambiguous date (both values <= 12), should prefer ISO format."""
    name = "pixel - 123 - 2025-03-05 14-30-00 - 2025-03-05 14-45-00"
    px, s, e = parse_times_from_filename(name)
    assert px == 123
    # Should interpret as March 5th (ISO format MM-DD)
    assert (s.year, s.month, s.day) == (2025, 3, 5)
    assert (e - s).total_seconds() == 15*60


def test_invalid_format():
    """Test that invalid format raises ValueError."""
    invalid_names = [
        "not a valid filename",
        "pixel - abc - 2025-19-08 10-19-59 - 2025-19-08 10-28-55",  # non-numeric pixel
        "pixel - 211 - invalid date - 2025-19-08 10-28-55",  # invalid date format
        ""
    ]
    
    for name in invalid_names:
        try:
            parse_times_from_filename(name)
            assert False, f"Should have raised ValueError for: {name}"
        except ValueError:
            pass  # Expected


def test_end_before_start():
    """Test that end time before start time raises ValueError."""
    # End time is before start time
    name = "pixel - 211 - 2025-19-08 10-28-55 - 2025-19-08 10-19-59"
    try:
        parse_times_from_filename(name)
        assert False, "Should have raised ValueError for end <= start"
    except ValueError as e:
        assert "End <= start" in str(e)


def test_with_extra_whitespace():
    """Test parsing with extra whitespace."""
    name = "  pixel  -  211  -  2025-19-08 10-19-59  -  2025-19-08 10-28-55  "
    px, s, e = parse_times_from_filename(name)
    assert px == 211
    assert (s.year, s.month, s.day, s.hour, s.minute, s.second) == (2025, 8, 19, 10, 19, 59)


def test_midnight_crossing():
    """Test parsing when time crosses midnight."""
    name = "pixel - 100 - 2025-19-08 23-55-00 - 2025-20-08 00-05-00"
    px, s, e = parse_times_from_filename(name)
    assert px == 100
    assert (s.year, s.month, s.day, s.hour, s.minute, s.second) == (2025, 8, 19, 23, 55, 0)
    assert (e.year, e.month, e.day, e.hour, e.minute, e.second) == (2025, 8, 20, 0, 5, 0)
    assert (e - s).total_seconds() == 10*60


if __name__ == "__main__":
    # Run tests manually
    test_functions = [
        test_dd_mm_variant,
        test_iso_variant,
        test_ambiguous_date,
        test_invalid_format,
        test_end_before_start,
        test_with_extra_whitespace,
        test_midnight_crossing
    ]
    
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")