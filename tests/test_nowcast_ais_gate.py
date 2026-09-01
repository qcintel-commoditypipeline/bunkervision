"""
Gate tests: alpha must be 0 when the AIS signal is not confirmed SIGNAL.

These tests verify the core invariant introduced in the hardening fix:
  calibrated_nowcast() never applies an AIS nudge unless signal_test()
  has returned a SIGNAL verdict (Pearson r > 0, p < 0.10, positive LOO skill).

Run: python -m pytest -q tests/
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

import ais_signal
import nowcast_model as nm


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fake_st(verdict: str, n: int = 0) -> dict:
    return {
        "verdict": verdict,
        "n_usable_pairs": n,
        "n_months_ais": n,
        "pairs": [],
        "tentative_at": "2027-01",
        "confident_at": "2027-07",
    }


def _fake_pairs(n: int, ais_dev_vals, resid_vals) -> pd.DataFrame:
    return pd.DataFrame({
        "year": [2026] * n,
        "month": list(range(1, n + 1)),
        "label": [f"2026-{i:02d}" for i in range(1, n + 1)],
        "clean_per_day": [10.0] * n,
        "coverage": [0.9] * n,
        "good": [True] * n,
        "ais_dev_pct": list(ais_dev_vals),
        "level_mt": [4_000_000.0] * n,
        "official_mt": [4_000_000.0] * n,
        "residual_pct": list(resid_vals),
    })


# ── live_coupling: alpha MUST be 0 for every non-SIGNAL verdict ──────────────

@pytest.mark.parametrize("verdict,n", [
    ("INSUFFICIENT", 0),
    ("DIRECTIONAL ONLY (n=1)", 1),
    ("DIRECTIONAL ONLY (n=2)", 2),
    ("EARLY READ - leaning WITH demand (n=3, low confidence)", 3),
    ("EARLY READ - leaning AGAINST demand (n=5, low confidence)", 5),
    ("NO SIGNAL (tentative)", 8),
    ("NO SIGNAL", 14),
])
def test_live_coupling_alpha_zero_when_not_signal(monkeypatch, verdict, n):
    """live_coupling must return alpha_hat=0.0 for every non-SIGNAL verdict."""
    monkeypatch.setattr(ais_signal, "signal_test", lambda port: _fake_st(verdict, n))
    c = ais_signal.live_coupling("singapore")
    assert not c["is_signal"], f"Expected is_signal=False for verdict={verdict!r}"
    assert c["alpha_hat"] == 0.0, f"Expected alpha_hat=0.0 for verdict={verdict!r}"


# ── live_coupling: alpha is fit from data when SIGNAL is confirmed ────────────

def test_live_coupling_fits_alpha_when_signal(monkeypatch):
    """When SIGNAL, live_coupling fits alpha from data, capped at MAX_ALPHA."""
    from config import NOWCAST_AIS_MAX_ALPHA

    monkeypatch.setattr(ais_signal, "signal_test", lambda port: _fake_st("SIGNAL (tentative)", 6))
    # Perfect positive correlation (ais_dev == residual in same units) → alpha_raw ~ 1.0
    vals = [10.0, -5.0, 8.0, -3.0, 12.0, -8.0]
    monkeypatch.setattr(ais_signal, "build_pairs", lambda port: _fake_pairs(6, vals, vals))

    c = ais_signal.live_coupling("singapore")
    assert c["is_signal"]
    assert c["alpha_hat"] == pytest.approx(NOWCAST_AIS_MAX_ALPHA, abs=1e-6), \
        "alpha_hat should be capped at NOWCAST_AIS_MAX_ALPHA for perfect correlation"


def test_live_coupling_fits_negative_alpha_for_negative_correlation(monkeypatch):
    """Alpha is fit with correct sign — negative correlation → negative alpha."""
    from config import NOWCAST_AIS_MAX_ALPHA

    monkeypatch.setattr(ais_signal, "signal_test", lambda port: _fake_st("SIGNAL", 14))
    # Perfect negative correlation
    ais_vals = [10.0, -5.0, 8.0, -3.0, 12.0, -8.0]
    resid_vals = [-10.0, 5.0, -8.0, 3.0, -12.0, 8.0]
    monkeypatch.setattr(ais_signal, "build_pairs", lambda port: _fake_pairs(6, ais_vals, resid_vals))

    c = ais_signal.live_coupling("singapore")
    assert c["is_signal"]
    assert c["alpha_hat"] == pytest.approx(-NOWCAST_AIS_MAX_ALPHA, abs=1e-6), \
        "alpha_hat should be negative for negative correlation"


# ── calibrated_nowcast: nowcast == level when not SIGNAL ─────────────────────

def test_calibrated_nowcast_equals_level_when_no_signal(monkeypatch):
    """calibrated_nowcast must produce nowcast == level_mt when alpha=0."""
    monkeypatch.setattr(
        ais_signal, "live_coupling",
        lambda port: {"is_signal": False, "alpha_hat": 0.0, "n_pairs": 0, "verdict": "INSUFFICIENT"},
    )
    monkeypatch.setattr(nm, "official_series", lambda *a: pd.DataFrame({"month": [], "y": []}))
    monkeypatch.setattr(nm, "level_forecast", lambda *a: 4_500_000.0)

    result = nm.calibrated_nowcast("singapore", 2026, 9)
    assert result["ais_alpha"] == 0.0
    assert result["nowcast_mt"] == result["level_mt"]
    assert result["ais_adjustment_pct"] == 0.0
    assert result["ais_signal_verdict"] == "INSUFFICIENT"


@pytest.mark.parametrize("verdict", [
    "DIRECTIONAL ONLY (n=2)",
    "EARLY READ - leaning WITH demand (n=4, low confidence)",
    "NO SIGNAL (tentative)",
    "NO SIGNAL",
])
def test_calibrated_nowcast_equals_level_for_all_non_signal_verdicts(monkeypatch, verdict):
    monkeypatch.setattr(
        ais_signal, "live_coupling",
        lambda port: {"is_signal": False, "alpha_hat": 0.0, "n_pairs": 4, "verdict": verdict},
    )
    monkeypatch.setattr(nm, "official_series", lambda *a: pd.DataFrame({"month": [], "y": []}))
    monkeypatch.setattr(nm, "level_forecast", lambda *a: 5_000_000.0)

    result = nm.calibrated_nowcast("singapore", 2026, 10)
    assert result["ais_alpha"] == 0.0
    assert result["nowcast_mt"] == result["level_mt"]
