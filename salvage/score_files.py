"""
Authoritative re-scoring by importing each model FILE from salvage/models/.

More faithful than exec'ing the returned code string: it runs exactly the file the
agent wrote and validated. Imports every salvage/models/<name>.py, grabs its
forecast_one, and re-runs it through btlib's walk-forward. Trusts no self-reported
number.
"""
from __future__ import annotations

import importlib
import json
import sys
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, "salvage")
sys.path.insert(0, "salvage/models")
import btlib  # noqa: E402

MODELS_DIR = Path("salvage/models")


def main() -> None:
    df = btlib.load_panel()
    rows = []
    # audit metadata from the workflow result (verdicts/overfit), keyed by name
    meta = {}
    try:
        payload = json.load(open("salvage/bakeoff_results.json"))
        for m in payload.get("models", []):
            meta[m["name"]] = m
    except Exception:
        pass

    files = sorted(p for p in MODELS_DIR.glob("*.py") if p.stem != "__init__")
    for p in files:
        name = p.stem
        rec = {"name": name,
               "audit_verdict": (meta.get(name) or {}).get("verdict"),
               "audit_leakage": (meta.get(name) or {}).get("leakage"),
               "overfit_risk": (meta.get(name) or {}).get("overfit_risk"),
               "self_mae_pct": (meta.get(name) or {}).get("mae_pct")}
        try:
            mod = importlib.import_module(name)
            importlib.reload(mod)
            fn = getattr(mod, "forecast_one")
            res = btlib.walk_forward(df, fn, min_train=36)
            mt = btlib.metrics(res)
            rec.update({"ok": mt.get("n", 0) > 0, "mae_pct": mt.get("mae_pct"),
                        "bias_pct": mt.get("bias_pct"), "median_abs_pct": mt.get("median_abs_pct"),
                        "rmse_pct": mt.get("rmse_pct"), "within_5pct": mt.get("within_5pct"),
                        "within_10pct": mt.get("within_10pct"), "worst_abs_pct": mt.get("worst_abs_pct"),
                        "n": mt.get("n"), "by_year": json.dumps(mt.get("by_year", {})), "error": ""})
        except Exception as e:
            rec.update({"ok": False, "mae_pct": None, "error": f"{type(e).__name__}: {e}"})
            print(f"  [FAIL] {name}: {rec['error']}", file=sys.stderr)
            traceback.print_exc()
        rows.append(rec)

    lb = pd.DataFrame(rows)
    for bname, bfn in [("BENCH_seasonal_naive", btlib.seasonal_naive),
                       ("BENCH_trailing12", btlib.last12_mean)]:
        mt = btlib.metrics(btlib.walk_forward(df, bfn, min_train=36))
        lb = pd.concat([lb, pd.DataFrame([{"name": bname, "ok": True, "mae_pct": mt["mae_pct"],
            "bias_pct": mt["bias_pct"], "median_abs_pct": mt["median_abs_pct"], "rmse_pct": mt["rmse_pct"],
            "within_5pct": mt["within_5pct"], "within_10pct": mt["within_10pct"],
            "worst_abs_pct": mt["worst_abs_pct"], "n": mt["n"], "by_year": json.dumps(mt["by_year"]),
            "audit_verdict": "benchmark", "audit_leakage": False}])], ignore_index=True)

    lb.to_csv("salvage/leaderboard.csv", index=False)
    ranked = lb[(lb["ok"] == True) & (lb["audit_leakage"] != True) & (lb["mae_pct"].notna())].sort_values("mae_pct")
    print("\n=== AUTHORITATIVE LEAGUE TABLE (re-scored from disk; sound, non-leaky) ===")
    cols = ["name", "mae_pct", "bias_pct", "median_abs_pct", "within_5pct", "within_10pct", "worst_abs_pct", "self_mae_pct", "by_year"]
    print(ranked[cols].to_string(index=False))
    failed = lb[lb["ok"] != True]
    if len(failed):
        print("\n--- failed ---")
        print(failed[["name", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()
