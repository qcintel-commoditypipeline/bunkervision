# BunkerVision — Demand-Engine Salvage Report

**Date:** 2026-06-16 · **Status:** prototype, behind `USE_CALIBRATED_NOWCAST=False` (nothing live changed) · **Mode:** test / not deployed / not committed pending review

---

## 1. Executive summary

The original demand engine did not *measure* bunker demand — it summed AIS event
*duration* × a fixed pump rate, and its monthly total only ever landed near the
official MPA figure because two large errors cancelled. The 16h cap we shipped broke
that cancellation and exposed a ~50% undercount. There is no way to *measure* tonnes
from AIS alone (the only physical stem-size signal, draught, is a dead static field).

**The salvage: stop measuring, start nowcasting.** The 76-month official MPA series is
strongly seasonal and trending — i.e. highly *forecastable*. We replace the broken
duration sum with a calibrated forecast of that series, optionally nudged by an AIS
leading indicator that earns statistical weight only as graded months accrue:

```
nowcast = level_model(month) × (1 + α · z_ais)
```

**Result (out-of-sample, 40 monthly folds, 2023–2026):**

| Model | MAE% | Bias% | within 5% | within 10% | worst |
|---|---|---|---|---|---|
| **Calibrated nowcast (recommended)** | **4.07** | **−0.42** | 26/40 | 38/40 | 14.5 |
| Seasonal baseline *(what the dashboard shows today)* | 8.35 | −8.28 | 12/40 | 24/40 | 18.2 |
| Trailing-12-month mean | 5.27 | −2.36 | 24/40 | 36/40 | 15.9 |
| Old duration engine *(April 2026, only graded month)* | — | **−52.0** | — | — | — |

The nowcast **halves the error** of the baseline the dashboard uses today and removes
its 8% downward bias. Against the old engine it is not close: April 2026 nowcast error
**+10.3%** vs the duration engine's **−52.0%**.

**Verdict:** don't kill the project — kill the *claim*. "We independently measure
Singapore bunker demand" is dead. "We publish a calibrated nowcast with a validated
~4% error band, ahead of the official figure" is real, defensible, and sellable.

---

## 2. Why the old engine was wrong (recap)

Established earlier this week, confirmed here:

- **85% of called tonnage came from events ≥6h alongside** — dwell / AIS-gap
  loitering, not pumping. The clean 30min–6h stems (real bunkering) were only ~14%.
- **The headline matched MPA by cancellation:** ~half the real stems detected ×
  duration-inflated long events ≈ the official total. Cap the long events (16h) and
  May collapses to ~2.1M (−53% vs the 4.548M official).
- **No independent stem-size signal exists.** The draught cross-check is 0/12,458 —
  AIS `MaximumStaticDraught` is a manually-set static field (barges report one constant
  value for thousands of pings). A 2,000 mt stem moves a Panamax ~1–2 cm, below AIS
  resolution. Duration is the *only* tonnage input, and it is a broken proxy.

Absolute tonnes therefore **cannot be measured** from this data. They can be nowcast.

---

## 3. The approach

### 3.1 Level model — forecast the official series

The official MPA TOTAL series (76 months, 2020-01 → 2026-04) has strong multiplicative
seasonality and an upward trend. The winning level model (see §4) is a **log-space blend
of a damped-additive ETS (85%) and a Theta model (15%)** with a lognormal bias
correction — chosen out-of-sample. Working in log space makes the error proportional
(aligned with the %-accuracy metric) and seasonality multiplicative, which fits how
bunker volume actually behaves. Ported verbatim to `nowcast_model.level_forecast` so the
live path is exactly what was validated.

### 3.2 AIS adjustment — the leading nudge (earns weight over time)

AIS's value is not tonnage; it is *timing*: "is this month running hotter or colder than
the seasonal norm?" `z_ais` is the standardized deviation of the **clean 30min–6h stem
rate** (never the phantom long-dwell tonnage) from its own trailing baseline. It is
applied with a shrinkage weight `α`:

- α = 0 until `NOWCAST_AIS_MIN_GRADED_MONTHS` (6) of AIS history have been graded
  against official figures — i.e. until there is out-of-sample evidence the signal helps.
- α then ramps linearly to a hard cap `NOWCAST_AIS_MAX_ALPHA` (±10%). **AIS informs,
  never dominates.**

**Today α = 0** (we have ~2 months of AIS history), so the nowcast *is* the level model.
This is deliberate: it degrades gracefully to the strong, validated baseline and cannot
be hijacked by a noisy 2-point AIS fit. This is also exactly the "back-fit May to 4.548"
trap, refused by design — we never fit AIS to a known answer.

**Why this matters for May:** the level model called May at 4.825M vs the 4.548M official
(+6.1%); April +10.3%. Both 2026 months ran *below* their seasonal norm. A working AIS
"running cold" signal (z < 0) is precisely what would have pulled these down — which is
the roadmap, not today's claim.

---

## 4. The bake-off (how the level model was chosen)

**Protocol.** Strict expanding-window walk-forward on the 76-month official series: to
predict month *t*, fit only on months < *t*. 40 out-of-sample folds (2023-01 →
2026-04). Metric: MAE% (proportional, matches how the figure is judged). Harness:
`salvage/btlib.py`. Toolchain: statsmodels + numpy (no sklearn — classical methods are
correct for 76 points; ML would overfit).

**Process.** A multi-agent run (`Workflow`) built 17 candidate models across 3 rounds of
loop-until-dry: each model was actually executed against the harness, then adversarially
audited for leakage / overfit / regime-fragility, with a synthesis agent proposing
challengers each round. **Every candidate was then re-scored independently from its own
source file** (`salvage/score_files.py`) — no agent-reported number is trusted on faith.
All 17 self-reported figures matched the independent re-score exactly.

**Full authoritative league table** (re-scored; sound, non-leaky; sorted by MAE%):

| Model | MAE% | Bias% | within5 | within10 | worst |
|---|---|---|---|---|---|
| log_space_leader_ensemble | 3.96 | −0.25 | 26 | 38 | 14.8 |
| theta_decomp_blend | 4.01 | 0.14 | 26 | 37 | 15.3 |
| theta_lag_ensemble | 4.01 | −0.08 | 27 | 38 | 15.0 |
| holt_winters | 4.04 | −0.47 | 26 | 39 | 14.2 |
| **log_theta_ets ← recommended** | **4.07** | **−0.42** | 26 | 38 | 14.5 |
| trimmed_mean_of_leaders | 4.10 | −1.16 | 26 | 39 | 15.2 |
| damped_theta_decomp_stack | 4.13 | −0.03 | 28 | 39 | 16.0 |
| log_theta_lag_hybrid | 4.17 | −0.47 | 27 | 38 | 14.0 |
| theta_ets | 4.18 | −0.53 | 24 | 38 | 14.6 |
| bias_constrained_inverse_mae_ensemble | 4.19 | −0.36 | 27 | 39 | 16.4 |
| decomp_regression_stack | 4.26 | 0.36 | 26 | 37 | 15.5 |
| lag_regression | 4.35 | −1.61 | 28 | 38 | 17.2 |
| seasonal_x_trend | 4.47 | −2.11 | 23 | 37 | 13.9 |
| ridge_seasonal_lags | 4.51 | −1.29 | 27 | 36 | 17.1 |
| sarima | 4.52 | 0.11 | 26 | 37 | 15.9 |
| stl_trend | 4.61 | 0.04 | 25 | 38 | 12.7 |
| seasonal_ols | 4.64 | −2.35 | 24 | 36 | 14.0 |
| bias_corrected_lag_regression | 4.73 | −0.58 | 23 | 35 | 14.6 |
| *BENCH trailing-12* | 5.27 | −2.36 | 24 | 36 | 15.9 |
| *BENCH seasonal-naive (dashboard today)* | 8.35 | −8.28 | 12 | 24 | 18.2 |

**Why `log_theta_ets` and not the 3.96% ensemble.** The top ~10 models cluster between
3.96% and 4.20% — **within out-of-sample noise over 40 folds**; the differences are not
statistically meaningful. `log_theta_ets` is the simplest near-top *single* model:
near-zero bias (−0.42%), low overfit risk, stable across every year (4.8/4.1/3.5/3.7),
and — unlike the ensembles — it does not refit 2–4 sub-models per fold, so it is faster
and far less fragile. Choosing it over a 0.1%-"better" 4-model stack is the conservative,
defensible engineering call. (`holt_winters`, 4.04%, is an equally valid single-model
alternative.)

---

## 5. What shipped (all OFF / isolated)

| Artifact | Purpose |
|---|---|
| `nowcast_model.py` | Prototype calibrated nowcast (level model + AIS shrinkage adjustment). Behind `USE_CALIBRATED_NOWCAST=False`. |
| `config.py` (new flags) | `USE_CALIBRATED_NOWCAST`, `NOWCAST_AIS_MIN_GRADED_MONTHS=6`, `NOWCAST_AIS_MAX_ALPHA=0.10`, `NOWCAST_LEVEL_MIN_TRAIN=30`. |
| `demand_model.running_demand_estimate` | Flag-gated branch; flag-off returns the dict **byte-identical**, flag-on swaps the nowcast into the headline. |
| `backtest_nowcast.py` | Reusable monthly OOS harness + `oos_table()` for the shadow API. |
| `/api/accuracy-backtest` + `/backtest` + `templates/backtest.html` | Read-only **shadow panel** — nowcast vs official vs old engine, side by side. Not in the nav; touches no live data. |
| `salvage/` | Bake-off scratch: panel CSVs, `btlib.py`, 18 model files, re-scorers, `leaderboard.csv`. |

**Nothing was committed or deployed.** Flipping `USE_CALIBRATED_NOWCAST=True` is the
single switch that makes the nowcast live on `/api/demand`; until then it is visible only
on `/backtest`.

---

## 6. Integrity / validation

- **No sacred table was written by any experiment.** All bake-off work ran on a copy
  (`salvage/bunkervision_backtest.db`); the live `port_bunker_sales`, `mpa_monthly`,
  `monthly_demand_estimates`, `estimate_grades` were read-only throughout.
- **Strict walk-forward OOS**, no look-ahead; every model independently re-scored from
  source. Leakage audit + re-score agree.
- **Live behavior unchanged**: flag-off path proven byte-identical; `/api/accuracy`
  verified still working.
- **Honest framing baked in**: this is a calibrated nowcast with a published error band,
  not a measurement. The shadow panel is labeled "SHADOW · not live".

---

## 7. The decisive test — does AIS earn its keep?

The level model forecasts *public* data. The entire question of whether there is a
*moat* is: **does AIS add signal on top of that forecast?** This is now wired as a
hard, falsifiable test (`ais_signal.py` / `python ais_signal.py`):

```
residual(t) = official(t) / level_forecast(t) − 1     # what the forecast MISSED
z_ais(t)    = standardized deviation of the clean 30min–6h stem rate vs its baseline
```

We test (1) `corr(z_ais, residual)` — does the signal point the right way — and (2)
**out-of-sample skill**: does `level × (1 + α̂·z)` beat `level` alone, with α̂ fit
leave-one-out (no peeking). If yes, AIS is the product. If after enough months it adds
nothing, we say so and the claim shrinks to "timely nowcast of public data".

The feature is a **relative deviation** (clean-stem rate vs its running mean), which
needs only ONE prior good month — so the first usable pair appears as soon as a second
full month closes (June, ~mid-July), not in the autumn. Every run is stamped with a
**confidence tier matched to n**, so it gives an honest read from day one and sharpens
monthly as the database grows:

| usable pairs (n) | tier | what it reports |
|---|---|---|
| 0 | INSUFFICIENT | nothing computable yet |
| 1–2 | DIRECTIONAL ONLY | sign agreement (did AIS and the surprise move the same way) — illustrative, not a correlation |
| 3–5 | EARLY READ (low confidence) | Spearman + OOS skill, "wide error bars, directional only" |
| 6–11 | SIGNAL / NO SIGNAL (tentative) | Pearson r + p, leave-one-out OOS skill |
| 12+ | SIGNAL / NO SIGNAL (confident) | same, with enough power to trust |

**Status today: INSUFFICIENT — 0 usable pairs.** AIS history began 2026-04-22; April is
a partial ramp month (coverage 0.3, excluded), May is the first good month (no prior to
deviate from). Both months ran *below* the level forecast (Apr −9.3%, May −5.7%), but
with n=0 that is not evidence of anything. The test **accrues automatically**:
`ais_signal.snapshot_previous_month()` logs each completed month to the additive
`ais_signal_log` table (wired into the monthly scheduler), and `signal_test()` turns the
growing pairs into a tiered verdict.

**Projection (~1 new pair/month, Singapore):** first DIRECTIONAL read ~mid-July 2026,
tentative SIGNAL/NO-SIGNAL ~**Dec 2026**, confident ~mid-2027. The data quality is
labelled at every step. This is the number that decides whether the AIS rig is worth its
cost for the demand product, or whether the product is the (still saleable) weeks-early
nowcast of the official figure. **It will be evidence, not opinion.**

## 8. Recommended next steps

1. **Review `/backtest`** (run locally: `USE_CALIBRATED_NOWCAST` can stay off; the page
   reads the nowcast directly). Confirm the story holds on the server once May's 4.548M
   official lands via Stackhero today.
2. **Reposition the product claim** to "calibrated nowcast / leading indicator with ~4%
   error band" — drop "measurement". Update marketing/UI copy accordingly.
3. **A/B in shadow for a few months.** Keep the flag off; let `/backtest` accumulate live
   months. The honest gate to flip it live: nowcast beats the current baseline on *new*
   months, not just history (it already does OOS).
4. **Let AIS earn its weight.** With ~2 months of AIS history, α=0. Re-evaluate the AIS
   nudge once ≥6 graded AIS months exist (~end 2026). Until then the value is the level
   model; AIS is upside, not yet booked.
5. **Calibrate per-port.** The model is port-agnostic; run `backtest_nowcast.py --port X`
   for Fujairah/Rotterdam/etc. to confirm the edge generalizes before broad rollout.

### How to reproduce
```
python backtest_nowcast.py --port singapore     # OOS league vs benchmarks
python nowcast_model.py singapore 2026 5         # a single month's nowcast
python ais_signal.py singapore                   # the AIS-vs-residual signal test (verdict)
python salvage/score_files.py                    # re-score all 18 bake-off models
# Shadow panel: run the app, visit /backtest
```
