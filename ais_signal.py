"""
The decisive test: does the AIS signal earn its keep - and a read that sharpens
month by month as the database grows.

The calibrated nowcast's *level* comes from forecasting the public official series
(seasonality + trend). The only thing AIS can add is a *leading nudge* on top - and
the honest question is whether it actually does. This module measures exactly that
and reports it with a confidence tier matched to how much data exists, so it gives a
crude DIRECTIONAL read from the very first computable month and tightens toward a
CONFIDENT verdict as months accrue.

    residual(t) = official(t) / level_forecast(t) - 1     # what the level model MISSED
    ais_dev(t)  = clean 30min-6h stem rate / its running mean - 1   # "running hot / cold"

If AIS leads, ais_dev(t) co-moves with residual(t): when AIS runs hot, the month
beats the seasonal+trend forecast. We report, scaled to n:
  * directional agreement  - how often sign(ais_dev) == sign(residual)
  * Spearman / Pearson corr - does the signal point the right way
  * leave-one-out OOS skill - does level x (1 + alpha_hat*ais_dev) beat level alone

Honest by construction: a relative-deviation feature needs only ONE prior good month
(vs a z-score's ~3), so the first pair appears as soon as a second full month closes.
Every read is stamped with n and a confidence tier; nothing is presented as more
certain than the data supports. Partial-coverage ramp months (e.g. April 2026) are
flagged and excluded from the fit but still shown.

Read-only except for the additive `ais_signal_log` table; never touches a sacred table.
"""
from __future__ import annotations

import calendar
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

import db
import nowcast_model as nm

try:
    from scipy.stats import pearsonr, spearmanr
except Exception:  # pragma: no cover
    pearsonr = spearmanr = None

# A month must cover at least this fraction of its calendar days (distinct active
# event-days) to be trusted as a comparable AIS observation. Excludes ramp months
# like April 2026 (system started ~22nd) from the FIT, while still logging them.
_MIN_COVERAGE = 0.6
# The relative-deviation feature needs only this many prior good months (the mean).
_MIN_BASELINE = 1
# Confidence tiers by number of usable (feature + residual, good-coverage) pairs.
_TIER_DIRECTIONAL = 1   # 1-2 pairs: sign agreement only, illustrative
_TIER_LOW = 3           # 3-5 pairs: early correlation, low confidence
_TIER_TENTATIVE = 6     # 6-11 pairs: tentative verdict
_TIER_CONFIDENT = 12    # 12+ pairs: confident verdict


def _ensure_log() -> None:
    db.execute("""
        CREATE TABLE IF NOT EXISTS ais_signal_log (
            port           TEXT,
            year           INTEGER,
            month          INTEGER,
            clean_stems    INTEGER,
            active_days    INTEGER,
            days_in_month  INTEGER,
            clean_per_day  REAL,
            coverage       REAL,
            logged_at      TIMESTAMP,
            PRIMARY KEY (port, year, month)
        )
    """)


def monthly_features(port: str = "singapore") -> pd.DataFrame:
    """Reconstruct per-month AIS clean-stem features from bunkering_events.
    The clean 30min-6h band only - never the phantom long-dwell tail."""
    rows = db.query("""
        SELECT YEAR(start_ts) AS y, MONTH(start_ts) AS m,
               COUNT(*) AS clean_stems,
               COUNT(DISTINCT CAST(start_ts AS DATE)) AS active_days
        FROM bunkering_events
        WHERE port = ? AND open = FALSE
          AND duration_min >= 30 AND duration_min < 360
        GROUP BY 1, 2 ORDER BY 1, 2
    """, [port])
    df = pd.DataFrame(rows, columns=["year", "month", "clean_stems", "active_days"])
    if df.empty:
        return df
    df["days_in_month"] = df.apply(lambda r: calendar.monthrange(int(r["year"]), int(r["month"]))[1], axis=1)
    df["clean_per_day"] = df["clean_stems"] / df["active_days"].clip(lower=1)
    df["coverage"] = df["active_days"] / df["days_in_month"]
    return df


def snapshot_previous_month(port: str = "singapore") -> bool:
    """Persist the just-completed month's features into ais_signal_log (idempotent).
    Intended to run on the monthly scheduler so the evidence base is durable."""
    _ensure_log()
    today = date.today()
    py, pm = (today.year - 1, 12) if today.month == 1 else (today.year, today.month - 1)
    feats = monthly_features(port)
    row = feats[(feats["year"] == py) & (feats["month"] == pm)]
    if row.empty:
        return False
    r = row.iloc[0]
    db.execute("""
        INSERT INTO ais_signal_log
            (port, year, month, clean_stems, active_days, days_in_month,
             clean_per_day, coverage, logged_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (port, year, month) DO UPDATE SET
            clean_stems=excluded.clean_stems, active_days=excluded.active_days,
            days_in_month=excluded.days_in_month, clean_per_day=excluded.clean_per_day,
            coverage=excluded.coverage, logged_at=excluded.logged_at
    """, [port, int(py), int(pm), int(r["clean_stems"]), int(r["active_days"]),
          int(r["days_in_month"]), float(r["clean_per_day"]), float(r["coverage"]),
          datetime.now(timezone.utc).isoformat()])
    return True


def _official(port: str, year: int, month: int) -> float | None:
    series = nm.official_series(port, 9999, 12)
    if series.empty:
        return None
    hit = series[(series["month"].dt.year == year) & (series["month"].dt.month == month)]
    return float(hit["y"].iloc[0]) if len(hit) else None


def build_pairs(port: str = "singapore") -> pd.DataFrame:
    """Per-month table of (ais_dev, level_forecast, official, residual_pct).
    ais_dev = clean_per_day / running-mean-of-prior-good-months - 1 (causal)."""
    feats = monthly_features(port).sort_values(["year", "month"]).reset_index(drop=True)
    if feats.empty:
        return feats

    out = []
    prior_rates: list[float] = []  # prior good-coverage clean_per_day, causal
    for _, r in feats.iterrows():
        y, m = int(r["year"]), int(r["month"])
        good = bool(r["coverage"] >= _MIN_COVERAGE)
        if len(prior_rates) >= _MIN_BASELINE:
            base = float(np.mean(prior_rates))
            dev = (r["clean_per_day"] / base - 1.0) if base > 0 else np.nan
        else:
            dev = np.nan
        level = nm.level_forecast(nm.official_series(port, y, m), m)
        official = _official(port, y, m)
        resid = (official / level - 1.0) if (official and level) else np.nan
        out.append({
            "year": y, "month": m, "label": f"{y}-{m:02d}",
            "clean_per_day": round(float(r["clean_per_day"]), 2),
            "coverage": round(float(r["coverage"]), 2), "good": good,
            "ais_dev_pct": None if pd.isna(dev) else round(float(dev) * 100, 2),
            "level_mt": None if not level else round(level, 0),
            "official_mt": official,
            "residual_pct": None if pd.isna(resid) else round(float(resid) * 100, 2),
        })
        if good:
            prior_rates.append(float(r["clean_per_day"]))
    return pd.DataFrame(out)


def _project_dates(port: str, n_now: int) -> dict:
    """Rough calendar projection of when each confidence tier is reached, assuming
    ~1 new usable Singapore month-pair accrues per month."""
    today = date.today()

    def plus(months: int) -> str:
        mm = today.month - 1 + months
        return f"{today.year + mm // 12}-{mm % 12 + 1:02d}"

    return {
        "tentative_at": plus(max(0, _TIER_TENTATIVE - n_now)),
        "confident_at": plus(max(0, _TIER_CONFIDENT - n_now)),
    }


def signal_test(port: str = "singapore") -> dict:
    """Run the AIS-vs-residual test and return a verdict stamped with a confidence
    tier matched to how much usable data exists. Sharpens as the DB grows."""
    pairs = build_pairs(port)
    usable = pairs[pairs["good"]].dropna(subset=["ais_dev_pct", "residual_pct"]) \
        if not pairs.empty else pairs
    n = len(usable)
    res = {"port": port, "n_months_ais": int(len(pairs)), "n_usable_pairs": int(n),
           "pairs": pairs.to_dict("records"), **_project_dates(port, n)}

    if n == 0:
        res.update(confidence="none", verdict="INSUFFICIENT",
                   detail="No usable (feature + residual, good-coverage) pairs yet - "
                          "needs a second full month to close so a relative deviation "
                          "can be computed. AIS history began 2026-04.")
        return res

    g = usable["ais_dev_pct"].to_numpy(float) / 100.0
    rr = usable["residual_pct"].to_numpy(float) / 100.0
    # Directional agreement: did AIS and the demand surprise move the same way?
    agree = int(np.sum(np.sign(g) == np.sign(rr)))
    res["directional_agreement"] = f"{agree}/{n}"

    # Correlations (need >= 2 points; rank corr is the robust early read).
    if n >= 2 and pearsonr is not None:
        try:
            res["spearman_r"] = round(float(spearmanr(g, rr)[0]), 3)
            pr, pp = pearsonr(g, rr)
            res["pearson_r"], res["pearson_p"] = round(float(pr), 3), round(float(pp), 3)
        except Exception:
            res["spearman_r"] = res["pearson_r"] = res["pearson_p"] = None

    # Leave-one-out OOS skill (need >= 4 to be meaningful).
    if n >= 4:
        base_e, nudge_e = [], []
        for i in range(n):
            tr = np.delete(np.arange(n), i)
            zt, rt = g[tr], rr[tr]
            var = float(np.var(zt))
            ahat = float(np.cov(zt, rt)[0, 1] / var) if var > 0 else 0.0
            ahat = max(min(ahat, 0.10), -0.10)
            base_e.append(abs(rr[i]))
            nudge_e.append(abs(rr[i] - ahat * g[i]))
        res["oos_skill_pct"] = round((np.mean(base_e) - np.mean(nudge_e)) / np.mean(base_e) * 100, 2) \
            if np.mean(base_e) else 0.0

    # Tiered verdict.
    lean = "with" if (res.get("spearman_r") or (agree - n / 2)) and \
        ((res.get("spearman_r") or 0) > 0 or agree > n / 2) else "against"
    if n < _TIER_LOW:           # 1-2 pairs
        res.update(confidence="directional",
                   verdict=f"DIRECTIONAL ONLY (n={n})",
                   detail=f"{agree}/{n} months moved the same way as AIS. Illustrative, "
                          "not yet a correlation - one or two points. Improves monthly.")
    elif n < _TIER_TENTATIVE:   # 3-5 pairs
        res.update(confidence="low",
                   verdict=f"EARLY READ - leaning {lean.upper()} demand (n={n}, low confidence)",
                   detail=f"Spearman {res.get('spearman_r')}, agreement {agree}/{n}, "
                          f"OOS skill {res.get('oos_skill_pct')}%. Wide error bars; directional only.")
    else:                       # 6+ pairs
        pr = res.get("pearson_r") or 0.0
        pp = res.get("pearson_p")
        skill = res.get("oos_skill_pct") or 0.0
        is_sig = pr > 0 and pp is not None and pp < 0.10 and skill > 0
        tier = "SIGNAL" if is_sig else "NO SIGNAL"
        conf = "confident" if n >= _TIER_CONFIDENT else "tentative"
        if conf == "tentative":
            tier += " (tentative)"
        res.update(confidence=conf, verdict=tier,
                   detail=f"corr(ais_dev, residual) r={pr} (p={pp}); OOS skill {skill}%; "
                          f"agreement {agree}/{n}. "
                          f"{'AIS is adding signal.' if is_sig else 'AIS not yet beating the forecast.'}")
    return res


def live_coupling(port: str = "singapore") -> dict:
    """Gated AIS coupling for the live nowcast — verdict + fitted alpha.

    Returns ``is_signal=True`` only when ``signal_test`` confirms a SIGNAL
    verdict (Pearson r > 0, p < 0.10, positive OOS skill). When the verdict
    is INSUFFICIENT / DIRECTIONAL / EARLY READ / NO SIGNAL, ``alpha_hat``
    is always 0.0 — safe default, no branch logic required by caller.

    When SIGNAL is confirmed, ``alpha_hat`` is the OLS slope of the level-model
    residual regressed on ``ais_dev`` (the relative clean-stem deviation) over
    all usable pairs, clipped to ``[-NOWCAST_AIS_MAX_ALPHA, +NOWCAST_AIS_MAX_ALPHA]``.
    Sign is fit from data — a negative true correlation produces a negative alpha.
    """
    from config import NOWCAST_AIS_MAX_ALPHA

    st = signal_test(port)
    verdict = st["verdict"]
    n = st["n_usable_pairs"]
    is_signal = verdict.startswith("SIGNAL")

    alpha_hat = 0.0
    if is_signal:
        pairs = build_pairs(port)
        usable = (
            pairs[pairs["good"]].dropna(subset=["ais_dev_pct", "residual_pct"])
            if not pairs.empty else pairs
        )
        if len(usable) >= 2:
            g = usable["ais_dev_pct"].to_numpy(float) / 100.0
            r = usable["residual_pct"].to_numpy(float) / 100.0
            var_g = float(np.var(g))
            if var_g > 0:
                alpha_raw = float(np.cov(g, r)[0, 1] / var_g)
                alpha_hat = float(
                    np.clip(alpha_raw, -NOWCAST_AIS_MAX_ALPHA, NOWCAST_AIS_MAX_ALPHA)
                )

    return {"is_signal": is_signal, "alpha_hat": alpha_hat, "n_pairs": n, "verdict": verdict}


def _print(port: str = "singapore") -> None:
    r = signal_test(port)
    W = 74
    print("=" * W)
    print(f"  AIS SIGNAL TEST - does AIS beat the seasonality+trend forecast? - {port}")
    print("=" * W)
    print(f"  Verdict: {r['verdict']}   [confidence: {r.get('confidence','none')}]")
    print(f"  {r['detail']}")
    print(f"\n  usable pairs now: {r['n_usable_pairs']}  ->  tentative ~{r['tentative_at']}, "
          f"confident ~{r['confident_at']}  (about 1 new month-pair/month)")
    print("\n  Per-month (ais_dev = AIS hot/cold; residual = official vs level forecast):")
    print(f"    {'month':8} {'cov':>5} {'clean/day':>10} {'ais_dev':>8} {'level':>10} {'official':>10} {'resid%':>8}")
    for p in r["pairs"]:
        flag = "" if p["good"] else "  (partial - excluded)"
        print(f"    {p['label']:8} {p['coverage']:>5} {p['clean_per_day']:>10} "
              f"{str(p['ais_dev_pct']):>8} {str(p['level_mt']):>10} {str(p['official_mt']):>10} "
              f"{str(p['residual_pct']):>8}{flag}")
    print("=" * W)


if __name__ == "__main__":
    import sys
    _print(sys.argv[1] if len(sys.argv) > 1 else "singapore")
