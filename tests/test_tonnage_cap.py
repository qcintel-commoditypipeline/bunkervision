"""Tonnage-cap math: estimated_mt = min(duration, EVENT_MAX_DURATION_MIN) * pump_rate."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT_PUMP_RATE_MT_MIN, EVENT_MAX_DURATION_MIN
from event_detector import capped_tonnage_mt


def test_short_event_unchanged():
    # A normal 2h event is unaffected by the cap
    assert capped_tonnage_mt(120, 3.5) == 420.0


def test_exactly_at_cap():
    assert capped_tonnage_mt(EVENT_MAX_DURATION_MIN, DEFAULT_PUMP_RATE_MT_MIN) == round(
        EVENT_MAX_DURATION_MIN * DEFAULT_PUMP_RATE_MT_MIN, 1
    )


def test_over_cap_is_capped():
    # An 18-day "event" must book the 16h cap, not a ~90k mt stem
    eighteen_days_min = 18 * 24 * 60
    assert capped_tonnage_mt(eighteen_days_min, 3.5) == EVENT_MAX_DURATION_MIN * 3.5
    assert capped_tonnage_mt(eighteen_days_min, 3.5) == 3360.0


def test_default_cap_is_16h():
    assert EVENT_MAX_DURATION_MIN == 960
