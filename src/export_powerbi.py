"""Export the star schema for the Power BI project (powerbi/TruckPdM.*) to powerbi/data/.

Reads data/processed/ (notebooks 02-04) and data/pdm.duckdb (built by src/build_dataset.py):

  dim_vehicle.csv           vehicle_id, split, Spec_0..Spec_7 (all 33,641 vehicles)
  dim_action.csv            planner action 0-4: label, window, sort order
  dim_cost_matrix.csv       COST[actual][predicted] with labels and cell type (correct/early/unnecessary/late/missed)
  fact_snapshot_scores.csv  test snapshot ("today's fleet", 5,045 trucks): calibrated p0..p4, frozen recommended
                            action, actual class, expected saving, priority, baseline actions, top-3 drivers and
                            the expected-cost terms L0..L4 / U0..U4 used by the DAX what-if measures (see below)
  shap_top_drivers.csv      long format: vehicle, driver rank, driver, effect (called-in test trucks)
  shap_global.csv           mean |SHAP| per input (notebook 03, validation)
  survival_curves.csv       Kaplan-Meier S(t) on a 5-step grid, overall and per spec group (training trucks)
  logrank_by_spec.csv       log-rank test per spec, Holm-corrected, repair share by t=300 per group
  cox_hazard_ratios.csv     Cox PH hazard ratios (specs + early-life usage), as in notebook 02
  reliability_key_numbers.csv  Weibull shape, repaired by 200/300/400, Cox concordance
  calibration.csv           predicted vs observed P(due) by age band and by risk bin, validation and test
  policy_comparison.csv     notebook 04: every policy on validation and test
  threshold_sweep.csv       notebook 04 (validation): miss-cost multiplier and workshop-capacity curves

Expected-cost terms. For action d, a truck's expected cost under miss-cost multiplier m is
    E[cost | d] = m * L_d + U_d,   L_d = sum_{k > d} p_k COST[k][d]  (late/missed part, scaled by m)
                                   U_d = sum_{k <= d} p_k COST[k][d] (early/unnecessary part)
so DAX can re-optimise the planner's decision for any m with plain arithmetic. replicate_dax()
below mirrors the DAX measures in Python; the asserts check that the dashboard's default view
(m = 1, no capacity limit) reproduces notebook 04's one-time test result exactly.

The survival, log-rank and Cox numbers repeat notebooks/02_reliability.ipynb (same pooling,
landmark and settings); asserts fail if they drift apart.

Run from the repo root:  python src/export_powerbi.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from costs import ACTIONS, CLASSES, COST  # noqa: E402
from make_features_sql import COUNTERS  # noqa: E402

PROC, OUT = ROOT / "data" / "processed", ROOT / "powerbi" / "data"
OUT.mkdir(parents=True, exist_ok=True)
SPEC_COLS = [f"Spec_{i}" for i in range(8)]
MIN_GROUP = 200
WINDOWS = {0: "no repair expected within 48 steps", 1: "repair in 24-48 steps", 2: "repair in 12-24 steps",
           3: "repair in 6-12 steps", 4: "repair within 6 steps"}
CLASS_LABELS = {0: "0 · none within 48", 1: "1 · in 24–48", 2: "2 · in 12–24", 3: "3 · in 6–12", 4: "4 · within 6"}


def write(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8")
    print(f"  {name}.csv  {len(df):>6,} rows")


con = duckdb.connect(str(ROOT / "data" / "pdm.duckdb"), read_only=True)

# ---------------------------------------------------------------- dimensions
dim_vehicle = con.execute(f"SELECT vehicle_id, split, {', '.join(SPEC_COLS)} FROM specifications "
                          "ORDER BY split, vehicle_id").df()
dim_action = pd.DataFrame({"action": CLASSES, "label": [ACTIONS[k] for k in CLASSES],
                           "urgency_window": ["none", "within 48 steps", "within 24 steps", "within 12 steps", "now"],
                           "sort_order": [5, 4, 3, 2, 1]})
cells = []
for a in CLASSES:
    for p in CLASSES:
        kind = ("correct" if a == p else "unnecessary call-in" if a == 0 else "missed repair" if p == 0
                else "late" if a > p else "early")
        cells.append({"actual": a, "actual_label": CLASS_LABELS[a], "predicted": p, "predicted_label": ACTIONS[p],
                      "cost": int(COST[a, p]), "cell_type": kind, "is_late_or_missed": int(a > p)})
dim_cost_matrix = pd.DataFrame(cells)

# ---------------------------------------------------------------- fact: test snapshot
svc = pd.read_csv(PROC / "test_service_list.csv")
P = svc[[f"p{k}" for k in CLASSES]].to_numpy()
age_rule = pd.read_csv(PROC / "fixed_age_rule.csv")
edges = age_rule.age_from.to_numpy()
age = svc.age_time_step.to_numpy()
fixed_age_action = age_rule["class"].to_numpy()[np.clip(np.digitize(age, edges) - 1, 0, len(edges) - 1)]
age_band_action = np.where((age >= 100) & (age < 250), 4, 0)

lower = np.tril(np.ones((5, 5), bool), -1)            # actual > predicted: late or missed
L = P @ np.where(lower, COST, 0)                        # (n, 5): L_d
U = P @ np.where(lower, 0, COST)                        # (n, 5): U_d
fact = pd.DataFrame({
    "vehicle_id": svc.vehicle_id, "age_time_step": age,
    **{f"p{k}": P[:, k] for k in CLASSES},
    "p_due_within_48": svc.p_due_within_48,
    "recommended_action": svc.recommended_action, "recommended_label": svc.recommended_label,
    "actual_class": svc.actual_class, "actual_label": svc.actual_class.map(CLASS_LABELS),
    "expected_saving_if_called_in": svc.expected_saving_if_called_in, "priority_rank": svc.priority_rank,
    "fixed_age_action": fixed_age_action, "age_band_action": age_band_action,
    **{f"L{d}": L[:, d] for d in CLASSES}, **{f"U{d}": U[:, d] for d in CLASSES},
    **{c: svc[c].fillna("") for c in ["driver_1", "driver_1_effect", "driver_2", "driver_2_effect",
                                      "driver_3", "driver_3_effect"]},
})


def replicate_dax(m: float = 1.0, slots_per_1000: float = 1000.0, frame: pd.DataFrame = fact) -> dict:
    """Python mirror of the DAX planner measures (see _Measures.tmdl)."""
    Lm, Um = frame[[f"L{d}" for d in CLASSES]].to_numpy(), frame[[f"U{d}" for d in CLASSES]].to_numpy()
    e = m * Lm + Um                                     # expected cost of each action
    best = e[:, 1:].min(axis=1)
    best_action = 1 + e[:, 1:].argmin(axis=1)           # ties -> least urgent, like SWITCH order in DAX
    saving = e[:, 0] - best
    n = len(frame)
    k = int(round(slots_per_1000 / 1000 * n))
    cands = np.sort(saving[saving > 0])[::-1]
    thr = np.inf if k <= 0 else (0.0 if k >= len(cands) else cands[k - 1])
    action = np.where((saving > 0) & (saving >= thr), best_action, 0)
    y = frame.actual_class.to_numpy()
    cost = np.where(y > action, m, 1) * COST[y, action]
    return {"cost_per_1000": 1000 * cost.mean(), "call_ins_per_1000": 1000 * (action > 0).mean(),
            "caught": int(((y > 0) & (action >= y)).sum()), "action": action}


ref = pd.read_csv(PROC / "policy_comparison.csv")
ref_test = ref[ref.split == "test"].set_index("policy")
default = replicate_dax()
assert (default["action"] == fact.recommended_action.to_numpy()).all(), "DAX mirror != frozen recommendation"
assert abs(default["cost_per_1000"] - ref_test.loc["Model policy (expected cost)", "cost_per_1000"]) < 0.05
assert default["caught"] == ref_test.loc["Model policy (expected cost)", "caught_in_time"]
for col, name in [("fixed_age_action", "Fixed-age rule (b)"), ("age_band_action", "Age bands (validation-tuned bar)")]:
    c = 1000 * COST[fact.actual_class, fact[col]].mean()
    assert abs(c - ref_test.loc[name, "cost_per_1000"]) < 0.05, name
print("DAX mirror reproduces notebook 04 on test (m = 1, no capacity limit): "
      f"{default['cost_per_1000']:,.1f} per 1,000 trucks")
for m, s in [(2.0, 1000), (1.0, 100), (1.0, 200)]:
    r = replicate_dax(m, s)
    print(f"  m = {m}, slots = {s}: cost per 1,000 {r['cost_per_1000']:,.0f}, call-ins {r['call_ins_per_1000']:,.0f}, "
          f"caught {r['caught']}")

drivers = []
for r in (1, 2, 3):
    d = svc.loc[svc[f"driver_{r}"].notna(), ["vehicle_id", f"driver_{r}", f"driver_{r}_effect"]]
    drivers.append(d.set_axis(["vehicle_id", "driver", "effect"], axis=1).assign(driver_rank=r))
shap_top = pd.concat(drivers).sort_values(["vehicle_id", "driver_rank"])[["vehicle_id", "driver_rank", "driver", "effect"]]
shap_global = pd.read_csv(PROC / "shap_importance.csv").rename(columns={"name": "input"})[["input", "mean_abs_shap"]]
shap_global["rank"] = np.arange(1, len(shap_global) + 1)

# ---------------------------------------------------------------- reliability (repeats notebook 02)
warnings.filterwarnings("ignore")
from lifelines import CoxPHFitter, KaplanMeierFitter, WeibullFitter  # noqa: E402
from lifelines.statistics import multivariate_logrank_test  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

surv = con.execute('''
    SELECT t.vehicle_id, t.length_of_study_time_step AS T, t.in_study_repair AS E, s.* EXCLUDE (split, vehicle_id)
    FROM train_tte t JOIN specifications s ON s.split = 'train' AND s.vehicle_id = t.vehicle_id
''').df()
for col in SPEC_COLS:
    counts = surv[col].value_counts()
    surv[f"{col}_g"] = surv[col].where(surv[col].isin(counts[counts >= MIN_GROUP].index), "Other")

grid = np.arange(0, 511, 5.0)
curves, lr_rows = [], []
kmf = KaplanMeierFitter().fit(surv["T"], surv["E"])
curves.append(pd.DataFrame({"spec": "Overall", "group": "All trucks", "time_step": grid,
                            "share_without_repair": kmf.survival_function_at_times(grid).values,
                            "trucks_at_risk": [(surv["T"] >= t).sum() for t in grid], "group_size": len(surv)}))
for col in SPEC_COLS:
    g = surv[f"{col}_g"]
    for k in g.value_counts().index:
        mask = g == k
        kf = KaplanMeierFitter().fit(surv["T"][mask], surv["E"][mask])
        at_risk = np.array([(surv["T"][mask] >= t).sum() for t in grid])
        keep = at_risk >= 20                              # don't draw curves on fewer than 20 trucks
        curves.append(pd.DataFrame({"spec": col, "group": k, "time_step": grid[keep],
                                    "share_without_repair": kf.survival_function_at_times(grid[keep]).values,
                                    "trucks_at_risk": at_risk[keep], "group_size": int(mask.sum())}))
    test = multivariate_logrank_test(surv["T"], g, surv["E"])
    by300 = {k: 1 - KaplanMeierFitter().fit(surv["T"][g == k], surv["E"][g == k]).survival_function_at_times(300).iloc[0]
             for k in g.unique()}
    lr_rows.append({"spec": col, "groups": g.nunique(), "chi2": test.test_statistic, "p_raw": test.p_value,
                    "lowest_group": min(by300, key=by300.get), "repaired_by_300_lowest": min(by300.values()),
                    "highest_group": max(by300, key=by300.get), "repaired_by_300_highest": max(by300.values())})
survival_curves = pd.concat(curves, ignore_index=True)
logrank = pd.DataFrame(lr_rows)
logrank["p_holm"] = multipletests(logrank.p_raw, method="holm")[1]
logrank["significant_5pct"] = np.where(logrank.p_holm < 0.05, "yes", "no")
logrank["p_holm_display"] = logrank.p_holm.map(lambda p: "< 1e-10" if p < 1e-10 else f"{p:.2g}")
assert logrank.set_index("spec").loc["Spec_5", "significant_5pct"] == "no"

wf = WeibullFitter().fit(surv["T"], surv["E"])
rho, rho_lo, rho_hi = wf.summary.loc["rho_", ["coef", "coef lower 95%", "coef upper 95%"]]
assert round(rho, 2) == 2.40, rho

LANDMARK, MIN_SPAN = 60, 10
cols = ", ".join(f'(l."{c}_0" - f."{c}_0") / (e.t1 - e.t0) AS rate_{c}' for c in COUNTERS)
early = con.execute(f'''
    WITH w AS (SELECT * FROM readouts WHERE split = 'train' AND time_step <= {LANDMARK}),
    e AS (SELECT vehicle_id, min(time_step) AS t0, max(time_step) AS t1 FROM w GROUP BY 1)
    SELECT e.vehicle_id, e.t1 - e.t0 AS early_span, {cols}
    FROM e JOIN w f ON f.vehicle_id = e.vehicle_id AND f.time_step = e.t0
           JOIN w l ON l.vehicle_id = e.vehicle_id AND l.time_step = e.t1
''').df()
con.close()
rate_cols = [f"rate_{c}" for c in COUNTERS]
cox_df = surv.merge(early, on="vehicle_id", how="left")
cox_df = cox_df[(cox_df.early_span >= MIN_SPAN) & cox_df[rate_cols].notna().all(axis=1)].copy()
usage_cols = []
for c in rate_cols:
    x = np.log1p(cox_df[c].clip(lower=0))
    cox_df[c.replace("rate_", "usage_")] = (x - x.mean()) / x.std()
    usage_cols.append(c.replace("rate_", "usage_"))
dummies = []
for col in SPEC_COLS:
    g = cox_df[f"{col}_g"]
    dummies.append(pd.get_dummies(g, prefix=col, dtype=float).drop(columns=f"{col}_{g.value_counts().index[0]}"))
X = pd.concat([cox_df[["T", "E"] + usage_cols]] + dummies, axis=1)
X = X.loc[:, X.std() > 0]
cph = CoxPHFitter(penalizer=0.01).fit(X, duration_col="T", event_col="E")
hr = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].copy()
hr.columns = ["hazard_ratio", "hr_low_95", "hr_high_95", "p_raw"]
hr["p_holm"] = multipletests(hr.p_raw, method="holm")[1]
hr = hr.reset_index().rename(columns={"covariate": "covariate"})
hr["kind"] = np.where(hr.covariate.str.startswith("usage_"), "early-life usage (per 1 SD of log rate)",
                      "spec category (vs largest category)")
hr["label"] = np.where(hr.covariate.str.startswith("usage_"),
                       "counter " + hr.covariate.str.replace("usage_", "", regex=False) + " usage",
                       hr.covariate.str.replace("_Cat", " = Cat", regex=False).str.replace("_Other", " = Other", regex=False))
hr["significant_5pct"] = np.where(hr.p_holm < 0.05, "yes", "no")
assert round(hr.set_index("covariate").loc["usage_427", "hazard_ratio"], 2) == 1.29

key = pd.DataFrame([
    {"metric": "weibull_shape", "label": "Weibull shape ρ", "value": rho,
     "note": f"95% CI {rho_lo:.2f}–{rho_hi:.2f}; > 1 = wear-out"},
    *[{"metric": f"repaired_by_{t}", "label": f"Repaired by time step {t}", "value": 1 - kmf.survival_function_at_times(t).iloc[0],
       "note": "Kaplan–Meier, training trucks"} for t in (200, 300, 400)],
    {"metric": "cox_concordance", "label": "Cox concordance (training)", "value": cph.concordance_index_,
     "note": "specs + early-life usage; optimistic (fitted data)"},
])

# ---------------------------------------------------------------- calibration
vs = pd.read_csv(PROC / "validation_scores.csv")
cal_src = pd.concat([
    pd.DataFrame({"split": "validation", "age": vs.time_step, "pred": 1 - vs.p0, "due": (vs.actual_class > 0).astype(int)}),
    pd.DataFrame({"split": "test", "age": fact.age_time_step, "pred": fact.p_due_within_48,
                  "due": (fact.actual_class > 0).astype(int)}),
])
AGE_BINS = [0, 100, 150, 200, 250, 300, 1e9]
AGE_LABELS = ["< 100", "100–150", "150–200", "200–250", "250–300", "≥ 300"]
rows = []
for split, d in cal_src.groupby("split"):
    d = d.copy()
    d["age_band"] = pd.cut(d.age, AGE_BINS, right=False, labels=AGE_LABELS)
    for i, (band, g) in enumerate(d.groupby("age_band", observed=True)):
        rows.append({"split": split, "view": "age band", "bin_label": band, "bin_order": i + 1,
                     "trucks": len(g), "predicted_share_due": g.pred.mean(), "observed_share_due": g.due.mean()})
    d["risk_bin"] = pd.qcut(d.pred.rank(method="first"), 8, labels=False)
    for i, g in d.groupby("risk_bin"):
        rows.append({"split": split, "view": "risk bin", "bin_label": f"bin {i + 1} of 8", "bin_order": i + 1,
                     "trucks": len(g), "predicted_share_due": g.pred.mean(), "observed_share_due": g.due.mean()})
calibration = pd.DataFrame(rows)

# ---------------------------------------------------------------- notebook 04 tables
policy_comparison = ref.rename(columns={"share_caught_in_time": "share_caught_in_time"})
sens = pd.read_csv(PROC / "miss_cost_sensitivity.csv")
cap = pd.read_csv(PROC / "capacity_curve.csv")
threshold_sweep = pd.concat([
    sens.assign(sweep="miss-cost multiplier", x=sens.miss_cost_multiplier)[["sweep", "x", "policy", "cost_per_1000", "share_caught_in_time"]],
    cap.assign(sweep="workshop capacity (call-ins per 1,000)", x=cap.K_per_1000)[["sweep", "x", "policy", "cost_per_1000", "share_caught_in_time"]],
], ignore_index=True)

# ---------------------------------------------------------------- write
print(f"\nwriting {OUT.relative_to(ROOT)}/")
for df, name in [(dim_vehicle, "dim_vehicle"), (dim_action, "dim_action"), (dim_cost_matrix, "dim_cost_matrix"),
                 (fact.round(6), "fact_snapshot_scores"), (shap_top, "shap_top_drivers"), (shap_global.round(5), "shap_global"),
                 (survival_curves.round(5), "survival_curves"), (logrank.round(6), "logrank_by_spec"),
                 (hr.round(5), "cox_hazard_ratios"), (key.round(5), "reliability_key_numbers"),
                 (calibration.round(5), "calibration"), (policy_comparison.round(4), "policy_comparison"),
                 (threshold_sweep.round(4), "threshold_sweep")]:
    write(df, name)
