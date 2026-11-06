# Project plan: truck predictive maintenance (Scania Component X)

Portfolio project #2 for the Operations / machine-health tracks (Volvo Group graduate
applications). It follows the same workflow as `sweden-ev-adoption`: Claude Code in the repo,
CLAUDE.md as the rulebook, SQL → Python notebooks → Power BI (PBIP/TMDL) → README case study.

## What changed from the original idea, and why

| Original idea | Changed to | Why |
|---|---|---|
| AI4I 2020 or NASA C-MAPSS | **SCANIA Component X** (Swedish National Data Service, CC BY 4.0) | Real fleet data from ~33,600 Scania heavy trucks, not a simulation. Swedish, heavy-truck, published 2024–25, so it's far less overused than AI4I/C-MAPSS. AI4I has no machine history over time, so it can't answer "which machine this week". |
| Invented cost assumptions | **Expert cost matrix that ships with the data** | Scania's experts priced a missed or late repair at 200–500 and an early or unnecessary call-in at 7–10. The trade-off is grounded, not invented. SEK and downtime appear only as labelled what-if parameters. |
| Failure-risk model + regression | **Survival analysis → risk model → cost-minimising decision rule** | Survival analysis (Kaplan–Meier, Weibull, Cox) is duration modelling, the same family as econometric duration models, so it plays to your MSc. The decision rule is what turns model scores into a service plan. |
| "Which machines this week" | **"Which trucks to call in, and how urgently"** (5 urgency classes) | The data's own labels are time-to-repair windows (6 / 12 / 24 / 48 time steps), which map onto planner actions: Call in now, Book within 12, Book within 24, Plan within 48, No action. |

**Honest limits to state up front:** variables and time are anonymized (time_step is operating
time, not days), and some values are perturbed. The project is about the *decision method* and
its cost impact, not about naming physical causes.

**Fallback dataset** if Scania's files ever become unavailable: Microsoft Azure Predictive
Maintenance (100 machines, hourly 2015 telemetry, maintenance + failure logs; synthetic).
It has real calendar dates, so "this week" would be literal.

## Business question
> Which trucks should the workshop call in for Component X, and how urgently, so the fleet's
> combined cost of breakdowns and early repairs is as low as possible? And how much does that
> save compared with run-to-failure and with a fixed-age rule?

## Numbers to beat (cost per 1,000 trucks)

| Policy | Validation | Test |
|---|---|---|
| Run-to-failure (predict 0 for everyone) | 11,375 | 11,120 |
| Call every truck in now (predict 4 for everyone) | 9,823 | 9,846 |
| **Fixed-age rule (baseline b)**: age thresholds tuned on validation, older = more urgent; ends up as "call in every truck older than 65" | 9,702 (CV 10,028) | 9,721 |
| Age bands, any direction (stricter age-only bar): call in trucks aged 100–250, no action otherwise | 8,109 (CV 8,354) | 9,150 |
| **Model policy**: gradient boosting + calibration with age term + expected-cost rule | **6,538** (cross-fitted) | **7,399** |
| Model policy without the age term (declared secondary check) | 7,294 (cross-fitted) | 7,600 |

The two naive rows are computed from the published class counts. The two age rows come from
notebook 02 (`data/processed/baselines_validation.csv`): in-sample on validation, with repeated
5-fold cross-validation over validation trucks in brackets. The model rows come from notebooks
03–04. **Test was evaluated once, in notebook 04's final section, with everything frozen.**
Bootstrap 95% intervals for the model's test saving: 1,214–3,665 vs calling everyone in,
1,094–3,529 vs the fixed-age rule.

A missed repair is so expensive that **calling every truck in beats doing nothing**. That makes a
good hook for the README, and it's the bar the model has to clear. Age alone barely helps: the
wear-out rule is no better than calling everyone in out of sample. On validation, older trucks
are *less* often due (the opposite of training), which only an age rule tuned on validation can
exploit. Models must beat 9,702 and should be compared with 8,109 too.

## Repo structure (same pattern as the EV repo)

```
CLAUDE.md                 rules + metric definitions (already written)
data/SOURCES.md           dataset, licence, citation, quirks (already written)
data/raw/                 9 CSVs from download_data.py (git-ignored)
data/pdm.duckdb           DuckDB database (git-ignored)
data/processed/           small tables (committed)
sql/01_load.sql           CSV -> DuckDB tables
sql/02_labels.sql         time-to-repair + urgency class per training readout
sql/03_features.sql       window-function features (deltas, rates, rolling, histogram summaries)
src/download_data.py      (already written)
src/build_dataset.py      runs the SQL, writes processed tables
src/export_powerbi.py     tables for Power BI -> powerbi/data/
src/viz.py                shared plot style
notebooks/01_eda.ipynb
notebooks/02_reliability.ipynb
notebooks/03_risk_model.ipynb
notebooks/04_maintenance_policy.ipynb
powerbi/TruckPdM.pbip
docs/figures/, docs/screenshots/, docs/dashboard.pdf
```

SQL gets a real role this time (the EV repo's sql/ folder is empty): labels and features are
built in DuckDB with window functions, which is close to how fleet data is handled in practice.

## Notebooks

| # | Question | Method | Key outputs |
|---|---|---|---|
| 01 EDA | What does the fleet and its data look like? | Readouts per truck, sampling irregularity, repair vs censored, spec mix, counter trajectories before repair vs healthy, histogram summaries | Data-quality table, "what changes before a repair" figure |
| 02 Reliability | How does Component X fail over its life? | Kaplan–Meier (overall, by Spec), log-rank tests, Weibull fit (shape > 1 = wear-out, so preventive repair is worth doing), Cox PH with specs and early-life usage rates, PH check | Survival curves, hazard ratios, **fixed-age rule baseline** chosen to minimise cost on validation |
| 03 Risk model | Given a truck's readouts so far, how close is it to repair? | Vehicle-level prediction points; group-by-vehicle CV; (a) multinomial logistic / discrete-time hazard (interpretable), (b) scikit-learn HistGradientBoostingClassifier (multiclass); calibration check; SHAP for drivers | Calibrated P(class 0..4) per truck, model comparison on validation |
| 04 Policy | What should the planner do, and what is it worth? | Bayes decision rule: pick the action with the lowest expected cost, Σ P(actual) × COST[actual][action]; compare with the two baselines; sensitivity to the cost ratio and to workshop capacity (top-K call-ins); **one-time test evaluation** | Policy table (cost per 1,000 trucks, repairs caught, unnecessary call-ins), sensitivity curves, test-set "fleet snapshot" service list |

## Power BI dashboard (5 pages)

1. **Overview**: fleet size, trucks flagged by action, cost per 1,000 trucks under each policy, % of repairs caught in time.
2. **Service plan**: the test-set snapshot as "today's fleet". A ranked truck list with recommended action, risk score, top 3 drivers and a spec slicer. This is the page the planner acts on.
3. **Cost trade-off**: what-if parameters for *SEK per cost unit*, *miss-cost multiplier* and *workshop slots*. Expected cost and call-ins update live through DAX (SUMX over trucks × cost matrix). Plus a threshold-sweep line chart.
4. **Reliability**: Kaplan–Meier curves by spec, Weibull shape, hazard-ratio chart.
5. **Model quality**: confusion matrix next to the cost matrix, calibration chart, baseline vs model.

Star schema: `dim_vehicle` (id, split, Spec_0–7), `dim_action` (class, label, window),
`fact_snapshot_scores` (vehicle, p0–p4, recommended action, actual class), `dim_cost_matrix`
(actual, predicted, cost), plus standalone tables `survival_curves`, `policy_comparison`,
`threshold_sweep`, `shap_top_drivers`.

## Timeline (~3 weeks, same pace as EV)

- **Week 1:** download, DuckDB build + SQL labels/features, notebook 01
- **Week 2:** notebooks 02 and 03
- **Week 3:** notebook 04, Power BI, README, LinkedIn post

---

## Claude Code prompts, step by step

Paste these one at a time in Claude Code from the repo root. Review and commit after each one,
as with the EV project. CLAUDE.md already holds the rules, so the prompts stay short.

**0. Set up**
```
Read CLAUDE.md, data/SOURCES.md and docs/PLAN.md. Create the venv, install requirements.txt,
run src/download_data.py, then git init and make the first commit. Tell me the file sizes and
row counts of what was downloaded, and flag anything that differs from SOURCES.md.
```

**1. DuckDB build + labels**
```
Write sql/01_load.sql and sql/02_labels.sql and src/build_dataset.py that runs them into
data/pdm.duckdb. 02 should compute, for every training readout, time to repair and the urgency
class as defined in CLAUDE.md (half-open windows; censored readouts within 48 steps of end of
observation get NULL). Validate: class counts per split, and check our label logic reproduces the
validation class distribution's shape when we take one random readout per training vehicle.
Commit.
```

**2. Feature SQL**
```
Write sql/03_features.sql: for each readout, using only readouts at or before it (window
functions over vehicle_id ORDER BY time_step), build: current counter levels, deltas and rates
per time_step since the previous readout and over the last 5 readouts, histogram summaries
(total count, weighted mean bin, share in the top 2 bins, change in mean bin over the last 5
readouts), number of readouts so far, and time_step itself. Explain each feature group in a
comment. Export a vehicle-level prediction table for validation/test (last readout) and a training
table. Add a test proving no feature uses a later readout. Commit.
```

**3. Notebook 01 EDA**
```
Create notebooks/01_eda.ipynb following CLAUDE.md. Cover: fleet and split sizes, readouts per
vehicle and sampling gaps, repaired vs censored, spec distributions by split, counter and
histogram trajectories in the 48 steps before repair vs matched healthy trucks, missingness.
Save figures to docs/figures/01_*.png. End with 3 findings that motivate notebooks 02–03. Commit.
```

**4. Notebook 02 Reliability**
```
Create notebooks/02_reliability.ipynb: Kaplan–Meier overall and by each Spec (log-rank tests,
correct for multiple tests), Weibull fit with interpretation of the shape parameter, Cox PH with
specs and early-life usage rates (check proportional hazards), then derive the fixed-age rule
baseline: map time_step to an urgency class from the survival curve, tune its thresholds on
validation using the cost matrix. Report its cost per 1,000 trucks next to the two naive
baselines in docs/PLAN.md. Commit.
```

**5. Notebook 03 Risk model**
```
Create notebooks/03_risk_model.ipynb. Build training prediction points that mimic validation
(explain the sampling/weighting choice first). Group-by-vehicle CV. Fit (a) multinomial logistic
regression with standardized features and (b) HistGradientBoostingClassifier (multiclass) with class weights. Compare on
validation: challenge cost using argmax AND using the expected-cost rule, macro-F1, per-class
recall. Check calibration and recalibrate if needed. SHAP for the chosen model, phrased with
anonymized variable names. Do not touch test. Save scores and drivers to data/processed/. Commit.
```

**6. Notebook 04 Maintenance policy**
```
Create notebooks/04_maintenance_policy.ipynb. Implement the expected-cost decision rule. Compare
run-to-failure, call-everyone-in, fixed-age rule and model policy on validation: cost per 1,000
trucks, repairs caught in time, unnecessary call-ins, call-ins per 1,000 trucks. Sensitivity:
miss-cost multiplier 0.25–4x and workshop capacity (top-K call-ins). Freeze everything, then
evaluate ONCE on test in a clearly marked final section. Translate to illustrative SEK/downtime
with labelled assumptions. Export the test snapshot service list. Commit.
```

**7. Power BI**
```
Write src/export_powerbi.py for the star schema in docs/PLAN.md, then build powerbi/TruckPdM.pbip
(TMDL model + PBIR report), same approach as the EV repo: DataFolder parameter, 5 pages as in
PLAN.md, what-if parameters for SEK per cost unit, miss-cost multiplier and workshop slots, with
DAX measures for expected cost and call-ins. Write the DAX with comments. Commit.
```

**8. README + LinkedIn**
```
Write README.md in the same structure as the EV repo: headline finding, 3 key insights with
numbers, recommendation for a fleet maintenance planner, tech stack badges, method table,
data-model diagram, reproduce steps, limitations (anonymized/perturbed data, time_step is not
calendar time, test labels cover only the last readout). Cite the dataset per SOURCES.md. Then
draft a LinkedIn post (under 200 words) aimed at Volvo Group / fleet uptime analytics roles.
```

**9. Check before publishing**
```
Rerun all notebooks top to bottom, check every number in README against notebook outputs, and
confirm test was only used in notebook 04's final section. List anything that doesn't match.
```
