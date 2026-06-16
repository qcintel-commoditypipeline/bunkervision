"""
Authoritative re-scoring of every bake-off candidate.

The workflow's agents self-report metrics; we trust NONE of them. This script takes
the workflow's returned `models` array (saved to salvage/bakeoff_results.json),
re-executes each model's `forecast_one_code` in a clean namespace, and re-runs it
through the SAME btlib walk-forward harness to produce the real league table.

Output:
  salvage/leaderboard.csv  — every model, recomputed OOS metrics, audit verdict
  prints the ranked table (sound, non-leaky models first)
"""
from __future__ import annotations

import json
import sys
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, "salvage")
import btlib  # noqa: E402

RESULTS = "salvage/bakeoff_results.json"


def _exec_model(code: str):
    """Exec a forecast_one_code string and return the forecast_one callable."""
    ns: dict = {}
    # Pre-import the libraries models are allowed to use.
    preamble = (
        "import numpy as np\n"
        "import pandas as pd\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "try:\n"
        "    import statsmodels.api as sm\n"
        "    from statsmodels.tsa.holtwinters import ExponentialSmoothing\n"
        "    from statsmodels.tsa.statespace.sarimax import SARIMAX\n"
        "    from statsmodels.tsa.seasonal import STL\n"
        "except Exception:\n"
        "    pass\n"
    )
    exec(preamble + "\n" + code, ns)
    fn = ns.get("forecast_one")
    if fn is None:
        raise ValueError("no forecast_one defined")
    return fn


def main() -> None:
    with open(RESULTS) as f:
        payload = json.load(f)
    models = payload.get("models", payload if isinstance(payload, list) else [])

    df = btlib.load_panel()
    rows = []
    for m in models:
        name = m.get("name", "?")
        rec = {
            "name": name, "family": m.get("family"), "round": m.get("round"),
            "audit_verdict": m.get("verdict"), "audit_leakage": m.get("leakage"),
            "overfit_risk": m.get("overfit_risk"),
            "self_mae_pct": m.get("mae_pct"),
        }
        try:
            fn = _exec_model(m["forecast_one_code"])
            res = btlib.walk_forward(df, fn, min_train=36)
            mt = btlib.metrics(res)
            rec.update({
                "ok": mt.get("n", 0) > 0,
                "mae_pct": mt.get("mae_pct"), "bias_pct": mt.get("bias_pct"),
                "median_abs_pct": mt.get("median_abs_pct"), "rmse_pct": mt.get("rmse_pct"),
                "within_5pct": mt.get("within_5pct"), "within_10pct": mt.get("within_10pct"),
                "worst_abs_pct": mt.get("worst_abs_pct"), "n": mt.get("n"),
                "by_year": json.dumps(mt.get("by_year", {})),
                "error": "",
            })
        except Exception as e:
            rec.update({"ok": False, "mae_pct": None, "error": f"{type(e).__name__}: {e}"})
            print(f"  [FAIL] {name}: {rec['error']}", file=sys.stderr)
        rows.append(rec)

    lb = pd.DataFrame(rows)

    # Add reference benchmarks computed here for an apples-to-apples baseline row.
    for bname, bfn in [("BENCH_seasonal_naive", btlib.seasonal_naive),
                       ("BENCH_trailing12", btlib.last12_mean)]:
        mt = btlib.metrics(btlib.walk_forward(df, bfn, min_train=36))
        lb = pd.concat([lb, pd.DataFrame([{
            "name": bname, "family": "benchmark", "ok": True,
            "mae_pct": mt["mae_pct"], "bias_pct": mt["bias_pct"],
            "median_abs_pct": mt["median_abs_pct"], "rmse_pct": mt["rmse_pct"],
            "within_5pct": mt["within_5pct"], "within_10pct": mt["within_10pct"],
            "worst_abs_pct": mt["worst_abs_pct"], "n": mt["n"],
            "by_year": json.dumps(mt["by_year"]), "audit_verdict": "benchmark",
            "audit_leakage": False,
        }])], ignore_index=True)

    lb.to_csv("salvage/leaderboard.csv", index=False)

    # Print ranked: sound models (ran ok, not leaky, kept) by MAE
    def sound(r):
        return bool(r.get("ok")) and not bool(r.get("audit_leakage")) and r.get("mae_pct") is not None
    ranked = lb[lb.apply(sound, axis=1)].sort_values("mae_pct")
    print("\n=== AUTHORITATIVE LEAGUE TABLE (sound, non-leaky; re-scored) ===")
    cols = ["name", "mae_pct", "bias_pct", "within_5pct", "within_10pct", "worst_abs_pct", "audit_verdict", "by_year"]
    print(ranked[cols].to_string(index=False))
    failed = lb[~lb.apply(lambda r: bool(r.get("ok")), axis=1)]
    if len(failed):
        print("\n--- failed to run / disqualified ---")
        print(failed[["name", "audit_verdict", "audit_leakage", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()
