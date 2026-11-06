"""Generate powerbi/TruckPdM.pbip: TMDL semantic model + PBIR report (5 pages).

Run after src/export_powerbi.py:   python src/build_pbip.py

This writes the project from scratch (deterministic IDs, so reruns give the same files). Once the
report has been edited in Power BI Desktop, don't rerun it: Desktop edits would be overwritten.

Design (docs/PLAN.md):
- Star schema from powerbi/data/*.csv, loaded through a DataFolder text parameter.
- What-if parameters as calculated tables (GENERATESERIES) + SELECTEDVALUE measures:
  SEK per cost unit (illustrative), miss-cost multiplier, workshop slots per 1,000 trucks.
- The planner measures re-optimise every truck's action live: expected cost of action d under
  miss-cost multiplier m is m * L_d + U_d (L/U precomputed per truck by export_powerbi.py,
  which also mirrors these measures in Python and asserts the default view = notebook 04).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PBI = ROOT / "powerbi"
DATA = PBI / "data"
NAME = "TruckPdM"
SM = PBI / f"{NAME}.SemanticModel"
RP = PBI / f"{NAME}.Report"
THEME_SRC = Path.home() / "Desktop/Projects/EV-Sweden/sweden-ev-adoption/powerbi/SwedenEV.Report/StaticResources/SharedResources/BaseThemes/Fluent2-CY26SU08.json"
DATA_FOLDER = str(DATA) + "\\"

BLUE, ORANGE, AQUA, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#6f6e69", "#0b0b0b", "#52514e"
CITATION = ("Data: SCANIA Component X dataset, Swedish National Data Service, "
            "https://doi.org/10.5878/bnh5-ka77, CC BY 4.0. Costs are relative units set by Scania experts; "
            "SEK figures are illustrative assumptions.")


def tag(*parts) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "truckpdm/" + "/".join(map(str, parts))))


def vid(*parts) -> str:
    return hashlib.md5("/".join(map(str, parts)).encode()).hexdigest()[:20]


def q(name: str) -> str:
    """Quote a TMDL/DAX object name when needed."""
    return f"'{name}'" if any(ch in name for ch in " -()/.,&") or name[0].isdigit() else name


# =============================================================================== semantic model
FORMATS = {  # column format strings
    "p_due_within_48": "0.0%", "p0": "0.000", "p1": "0.000", "p2": "0.000", "p3": "0.000", "p4": "0.000",
    "expected_saving_if_called_in": "#,0.0", "age_time_step": "#,0", "share_without_repair": "0.0%",
    "predicted_share_due": "0.0%", "observed_share_due": "0.0%", "hazard_ratio": "0.00", "hr_low_95": "0.00",
    "hr_high_95": "0.00", "cost_per_1000": "#,0", "share_caught_in_time": "0.0%", "mean_abs_shap": "0.000",
    "chi2": "#,0.0", "repaired_by_300_lowest": "0.0%", "repaired_by_300_highest": "0.0%",
}
SORT_BY = {("dim_action", "label"): "sort_order", ("fact_snapshot_scores", "recommended_label"): "recommended_action",
           ("fact_snapshot_scores", "actual_label"): "actual_class", ("dim_cost_matrix", "actual_label"): "actual",
           ("dim_cost_matrix", "predicted_label"): "predicted", ("calibration", "bin_label"): "bin_order"}
HIDDEN = {("fact_snapshot_scores", f"{lu}{d}") for lu in "LU" for d in range(5)}
DESCRIPTIONS = {
    "fact_snapshot_scores": "Test snapshot (notebook 04): one row per test truck, scored once with the frozen model.",
    "dim_cost_matrix": "IDA 2024 challenge cost matrix COST[actual][predicted], relative units set by Scania experts.",
}
CSV_TABLES = ["dim_vehicle", "dim_action", "dim_cost_matrix", "fact_snapshot_scores", "shap_top_drivers", "shap_global",
              "survival_curves", "logrank_by_spec", "cox_hazard_ratios", "reliability_key_numbers", "calibration",
              "policy_comparison", "threshold_sweep"]


def csv_table(name: str) -> str:
    df = pd.read_csv(DATA / f"{name}.csv")
    lines = []
    if name in DESCRIPTIONS:
        lines.append(f"/// {DESCRIPTIONS[name]}")
    lines += [f"table {name}", f"\tlineageTag: {tag(name)}", ""]
    m_types = []
    for c in df.columns:
        if pd.api.types.is_integer_dtype(df[c]):
            dt, mt = "int64", "Int64.Type"
        elif pd.api.types.is_float_dtype(df[c]):
            dt, mt = "double", "type number"
        else:
            dt, mt = "string", "type text"
        m_types.append(f'{{"{c}", {mt}}}')
        lines.append(f"\tcolumn {q(c)}")
        lines.append(f"\t\tdataType: {dt}")
        if (name, c) in HIDDEN:
            lines.append("\t\tisHidden")
        if c in FORMATS:
            lines.append(f"\t\tformatString: {FORMATS[c]}")
        elif dt == "int64":
            lines.append("\t\tformatString: 0")
        lines.append(f"\t\tlineageTag: {tag(name, c)}")
        lines.append("\t\tsummarizeBy: none")
        lines.append(f"\t\tsourceColumn: {c}")
        if (name, c) in SORT_BY:
            lines.append(f"\t\tsortByColumn: {q(SORT_BY[(name, c)])}")
        lines += ["", "\t\tannotation SummarizationSetBy = Automatic", ""]
    lines += [
        f"\tpartition {name} = m",
        "\t\tmode: import",
        "\t\tsource =",
        "\t\t\t\tlet",
        f'\t\t\t\t\tSource = Csv.Document(File.Contents(DataFolder & "{name}.csv"), [Delimiter=",", Columns={len(df.columns)}, Encoding=65001, QuoteStyle=QuoteStyle.Csv]),',
        '\t\t\t\t\t#"Promoted Headers" = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),',
        f'\t\t\t\t\t#"Changed Type" = Table.TransformColumnTypes(#"Promoted Headers", {{{", ".join(m_types)}}}, "en-US")',
        "\t\t\t\tin",
        '\t\t\t\t\t#"Changed Type"',
        "",
        "\tannotation PBI_ResultType = Table",
        "",
    ]
    return "\n".join(lines)


def measure(name: str, dax: str, fmt: str, folder: str, doc: str, hidden: bool = False) -> str:
    dax = dax.strip("\n")
    out = [f"\t/// {doc}"]
    if "\n" in dax:
        out.append(f"\tmeasure {q(name)} =")
        out += ["\t\t\t" + ln for ln in dax.splitlines()]
    else:
        out.append(f"\tmeasure {q(name)} = {dax}")
    out.append(f"\t\tformatString: {fmt}")
    if hidden:
        out.append("\t\tisHidden")
    out.append(f"\t\tdisplayFolder: {folder}")
    out.append(f"\t\tlineageTag: {tag('measure', name)}")
    return "\n".join(out) + "\n"


def parameter_table(name: str, series: str, default: str, fmt: str, doc: str, dtype: str) -> str:
    vm = f"{name} Value"
    return "\n".join([
        f"/// What-if parameter: {doc}",
        f"table {q(name)}",
        f"\tlineageTag: {tag(name)}",
        "",
        measure(vm, f"SELECTEDVALUE ( {q(name)}[{name}], {default} )", fmt, "Parameters",
                f"Selected value of '{name}' ({default} when nothing or several values are selected).").rstrip("\n"),
        "",
        f"\tcolumn {q(name)}",
        f"\t\tdataType: {dtype}",
        f"\t\tformatString: {fmt}",
        f"\t\tlineageTag: {tag(name, 'column')}",
        "\t\tsummarizeBy: none",
        "\t\tsourceColumn: [Value]",
        "",
        "\t\textendedProperty ParameterMetadata =",
        "\t\t\t\t{",
        '\t\t\t\t  "version": 0',
        "\t\t\t\t}",
        "",
        "\t\tannotation SummarizationSetBy = User",
        "",
        f"\tpartition {q(name)} = calculated",
        "\t\tmode: import",
        f"\t\tsource = {series}",
        "",
        f"\tannotation PBI_Id = {vid(name)}",
        "",
    ])


POLICIES = [("Risk-based plan (model)", 1), ("Age bands (validation-tuned)", 2), ("Fixed-age rule (b)", 3),
            ("Call everyone in now", 4), ("Run-to-failure", 5)]
policy_table = "\n".join([
    "/// Axis for comparing policies side by side (Overview page); values come from [Policy Cost per 1000].",
    "table Policy",
    f"\tlineageTag: {tag('Policy')}",
    "",
    "\tcolumn Policy",
    "\t\tdataType: string",
    f"\t\tlineageTag: {tag('Policy', 'Policy')}",
    "\t\tsummarizeBy: none",
    "\t\tisNameInferred",
    "\t\tsourceColumn: [Policy]",
    "\t\tsortByColumn: Order",
    "",
    "\t\tannotation SummarizationSetBy = Automatic",
    "",
    "\tcolumn Order",
    "\t\tdataType: int64",
    "\t\tformatString: 0",
    f"\t\tlineageTag: {tag('Policy', 'Order')}",
    "\t\tsummarizeBy: none",
    "\t\tisNameInferred",
    "\t\tsourceColumn: [Order]",
    "",
    "\t\tannotation SummarizationSetBy = Automatic",
    "",
    "\tpartition Policy = calculated",
    "\t\tmode: import",
    '\t\tsource = DATATABLE ( "Policy", STRING, "Order", INTEGER, { '
    + ", ".join(f'{{ "{p}", {o} }}' for p, o in POLICIES) + " } )",
    "",
])

F = "fact_snapshot_scores"
# Per-truck planner logic, shared by every planner measure. Needs VAR m (miss-cost multiplier) and
# VAR thr (capacity threshold on expected saving) in the enclosing scope, and a row context on F.
ROW_LOGIC = f"""
-- expected cost of each action d for this truck: m * L_d + U_d
VAR e0 = m * {F}[L0]
VAR e1 = m * {F}[L1] + {F}[U1]
VAR e2 = m * {F}[L2] + {F}[U2]
VAR e3 = m * {F}[L3] + {F}[U3]
VAR e4 = m * {F}[L4] + {F}[U4]
VAR best = MIN ( MIN ( e1, e2 ), MIN ( e3, e4 ) )      -- cheapest action other than 'No action'
VAR saving = e0 - best                                 -- expected saving of acting at all
-- act only if acting is cheaper AND the truck is within workshop capacity (saving >= threshold);
-- among equally cheap actions take the least urgent one
VAR chosen = IF ( saving > 0 && saving >= thr, SWITCH ( TRUE (), e1 = best, 1, e2 = best, 2, e3 = best, 3, 4 ), 0 )
VAR y = {F}[actual_class]"""

HEADER = """VAR m = [Miss Cost Multiplier Value]
VAR thr = [Capacity Saving Threshold]"""


def planner(body: str, iterator: str = "SUMX", pre: str = "") -> str:
    """Wrap the per-truck logic: <iterator> over the fact table, returning <body> per truck."""
    lines = ROW_LOGIC.strip("\n").splitlines() + (pre.splitlines() if pre else []) + [f"RETURN {body}"]
    inner = "\n".join("\t\t" + ln for ln in lines)
    return f"{HEADER}\nRETURN\n\t{iterator} (\n\t\t{F},\n{inner}\n\t)"


def indent(text: str, n: int = 1) -> str:
    return "\n".join("\t" * n + ln for ln in text.splitlines())


def realized(action_expr: str) -> str:
    return f"""VAR m = [Miss Cost Multiplier Value]
VAR total =
	SUMX (
		{F},
		VAR y = {F}[actual_class]
		VAR a = {action_expr}
		VAR c = LOOKUPVALUE ( dim_cost_matrix[cost], dim_cost_matrix[actual], y, dim_cost_matrix[predicted], a )
		RETURN IF ( y > a, m * c, c )      -- late/missed cells scaled by the miss-cost multiplier
	)
RETURN DIVIDE ( total, [Trucks] ) * 1000"""


COST_LOOKUP = "VAR c = LOOKUPVALUE ( dim_cost_matrix[cost], dim_cost_matrix[actual], y, dim_cost_matrix[predicted], chosen )"
COST_BODY = "IF ( y > chosen, m * c, c )      -- late/missed cells scaled by the miss-cost multiplier"

MEASURES = [
    # ---- fleet
    ("Trucks", f"COUNTROWS ( {F} )", "#,0", "Fleet", "Trucks in the test snapshot (current filters)."),
    ("Repairs Due", f"CALCULATE ( [Trucks], {F}[actual_class] > 0 )", "#,0", "Fleet",
     "Trucks whose Component X repair falls within the next 48 time steps (actual class 1-4)."),
    # ---- planner helpers
    ("Capacity Saving Threshold", f"""VAR m = [Miss Cost Multiplier Value]
VAR fleet = CALCULATE ( COUNTROWS ( {F} ), ALLSELECTED ( {F} ) )
VAR K = ROUND ( [Workshop Slots Value] / 1000 * fleet, 0 )    -- call-ins allowed in the selected fleet
VAR scored =
	ADDCOLUMNS (
		CALCULATETABLE ( {F}, ALLSELECTED ( {F} ) ),
		"@saving",
			m * {F}[L0]
				- MIN (
					MIN ( m * {F}[L1] + {F}[U1], m * {F}[L2] + {F}[U2] ),
					MIN ( m * {F}[L3] + {F}[U3], m * {F}[L4] + {F}[U4] )
				)
	)
VAR candidates = FILTER ( scored, [@saving] > 0 )                -- trucks where acting pays at all
RETURN
	SWITCH (
		TRUE (),
		K <= 0, 1E+300,                                              -- no slots: nobody is called in
		K >= COUNTROWS ( candidates ), 0,                            -- enough slots: no limit
		MINX ( TOPN ( K, candidates, [@saving], DESC ), [@saving] )  -- saving of the K-th best truck
	)""", "#,0.00", "Planner (helpers)",
     "Smallest expected saving that still gets a workshop slot: trucks are ranked by expected saving and only the top K (= slots per 1,000 x selected fleet) are acted on.", True),
    # ---- planner (live)
    ("Planner Cost", planner(COST_BODY, pre=COST_LOOKUP), "#,0", "Planner",
     "Realized cost (relative units) of the live planner policy on the test snapshot: each truck gets the action with the lowest expected cost under the selected miss-cost multiplier, limited by workshop slots; costed against its actual class."),
    ("Planner Cost per 1000", "DIVIDE ( [Planner Cost], [Trucks] ) * 1000", "#,0", "Planner",
     "Planner cost per 1,000 trucks (headline metric)."),
    ("Planner Expected Cost per 1000",
     "VAR expected =\n" + indent(planner("IF ( chosen = 0, e0, best )   -- expected cost of the chosen action")) +
     "\nRETURN DIVIDE ( expected, [Trucks] ) * 1000", "#,0",
     "Planner", "The model's own forecast of the planner cost per 1,000 trucks (expected cost of the chosen actions, no actual classes used)."),
    ("Planner Call-ins", planner("IF ( chosen > 0, 1, 0 )"), "#,0", "Planner",
     "Trucks the planner acts on (any action other than 'No action')."),
    ("Planner Call-ins per 1000", "DIVIDE ( [Planner Call-ins], [Trucks] ) * 1000", "#,0", "Planner",
     "Call-ins per 1,000 trucks."),
    ("Planner Repairs Caught", planner("IF ( y > 0 && chosen >= y, 1, 0 )"), "#,0", "Planner",
     "Due repairs caught in time: actual class 1-4 and an action at least as urgent as needed."),
    ("Planner Share Caught", "DIVIDE ( [Planner Repairs Caught], [Repairs Due] )", "0.0%", "Planner",
     "Share of due repairs caught in time."),
    ("Planner Unnecessary Call-ins", planner("IF ( y = 0 && chosen > 0, 1, 0 )"), "#,0", "Planner",
     "Call-ins of trucks with no repair due within 48 steps."),
    ("Planner Action", f"IF (\n\tHASONEVALUE ( {F}[vehicle_id] ),\n" + indent(planner("chosen", iterator="MAXX")) + "\n)", "0",
     "Planner", "Live planner action (0-4) for a single truck; blank on totals."),
    ("Planner Action Label", "VAR a = [Planner Action]\nRETURN IF ( ISBLANK ( a ), BLANK (), LOOKUPVALUE ( dim_action[label], dim_action[action], a ) )",
     "@", "Planner", "Live planner action as text, e.g. 'Call in now'."),
    # ---- baselines
    ("Run-to-failure Cost per 1000", realized("0"), "#,0", "Baselines", "Cost per 1,000 trucks if no truck is called in."),
    ("Call-everyone Cost per 1000", realized("4"), "#,0", "Baselines", "Cost per 1,000 trucks if every truck is called in now."),
    ("Fixed-age Rule Cost per 1000", realized(f"{F}[fixed_age_action]"), "#,0", "Baselines",
     "Cost per 1,000 trucks under baseline (b), the fixed-age rule from notebook 02 (call in every truck older than 65)."),
    ("Age-band Rule Cost per 1000", realized(f"{F}[age_band_action]"), "#,0", "Baselines",
     "Cost per 1,000 trucks under the validation-tuned age bands (call in trucks aged 100-250)."),
    # ---- savings
    ("Saving vs Call-everyone per 1000", "[Call-everyone Cost per 1000] - [Planner Cost per 1000]", "#,0", "Savings",
     "Planner saving vs calling every truck in, cost units per 1,000 trucks."),
    ("Saving vs Fixed-age per 1000", "[Fixed-age Rule Cost per 1000] - [Planner Cost per 1000]", "#,0", "Savings",
     "Planner saving vs the fixed-age rule, cost units per 1,000 trucks."),
    ("Planner Cost SEK per 1000", "[Planner Cost per 1000] * [SEK per Cost Unit Value]", "#,0 \"SEK\"", "Savings",
     "Planner cost per 1,000 trucks in SEK. ILLUSTRATIVE: SEK per cost unit is an assumption (what-if parameter), not data."),
    ("Saving vs Call-everyone SEK per 1000", "[Saving vs Call-everyone per 1000] * [SEK per Cost Unit Value]", "#,0 \"SEK\"",
     "Savings", "Planner saving vs calling everyone in, SEK per 1,000 trucks. ILLUSTRATIVE (assumed SEK per cost unit)."),
    ("Policy Cost per 1000", """SWITCH (
	SELECTEDVALUE ( Policy[Policy] ),
	"Risk-based plan (model)", [Planner Cost per 1000],
	"Age bands (validation-tuned)", [Age-band Rule Cost per 1000],
	"Fixed-age rule (b)", [Fixed-age Rule Cost per 1000],
	"Call everyone in now", [Call-everyone Cost per 1000],
	"Run-to-failure", [Run-to-failure Cost per 1000]
)""", "#,0", "Savings", "Cost per 1,000 trucks for the policy on the Policy axis."),
    # ---- reliability
    ("Share Without Repair", "AVERAGE ( survival_curves[share_without_repair] )", "0.0%", "Reliability",
     "Kaplan-Meier share of training trucks still on their original Component X."),
    ("Weibull Shape", 'CALCULATE ( MAX ( reliability_key_numbers[value] ), reliability_key_numbers[metric] = "weibull_shape" )',
     "0.00", "Reliability", "Weibull shape rho; above 1 means the hazard rises with age (wear-out)."),
    ("Repaired by 300", 'CALCULATE ( MAX ( reliability_key_numbers[value] ), reliability_key_numbers[metric] = "repaired_by_300" )',
     "0.0%", "Reliability", "Share of trucks repaired by time step 300 (Kaplan-Meier)."),
    ("Repaired by 400", 'CALCULATE ( MAX ( reliability_key_numbers[value] ), reliability_key_numbers[metric] = "repaired_by_400" )',
     "0.0%", "Reliability", "Share of trucks repaired by time step 400 (Kaplan-Meier)."),
    ("Hazard Ratio", "AVERAGE ( cox_hazard_ratios[hazard_ratio] )", "0.00", "Reliability",
     "Cox hazard ratio (1 = no effect)."),
    # ---- model quality
    ("Predicted Share Due", "AVERAGE ( calibration[predicted_share_due] )", "0.0%", "Model quality",
     "Mean predicted P(repair due within 48 steps) in the bin."),
    ("Observed Share Due", "AVERAGE ( calibration[observed_share_due] )", "0.0%", "Model quality",
     "Observed share of trucks with a repair due within 48 steps in the bin."),
    ("Cost Matrix Cell", "SUM ( dim_cost_matrix[cost] )", "#,0", "Model quality", "Cost of one actual/predicted cell."),
    ("Validation Cost per 1000", 'CALCULATE ( AVERAGE ( policy_comparison[cost_per_1000] ), policy_comparison[split] = "validation" )',
     "#,0", "Model quality", "Notebook 04: cost per 1,000 trucks on validation (model: cross-fitted)."),
    ("Test Cost per 1000", 'CALCULATE ( AVERAGE ( policy_comparison[cost_per_1000] ), policy_comparison[split] = "test" )',
     "#,0", "Model quality", "Notebook 04: cost per 1,000 trucks on test (evaluated once)."),
]


# Words DAX rejects as variable names (found the hard way: 'VAR action' is a syntax error).
DAX_RESERVED_VARS = {"action", "column", "table", "measure", "define", "evaluate", "order", "by", "start", "at",
                     "return", "var", "true", "false", "not", "and", "or", "in", "asc", "desc"}


def check_var_names(name: str, dax: str) -> None:
    import re
    for v in re.findall(r"\bVAR\s+([A-Za-z_][A-Za-z0-9_]*)", dax):
        assert v.lower() not in DAX_RESERVED_VARS, f"measure [{name}]: VAR name '{v}' is reserved in DAX"


def measures_table() -> str:
    lines = ["table _Measures", f"\tlineageTag: {tag('_Measures')}", ""]
    for name, dax, fmt, folder, doc, *hidden in MEASURES:
        check_var_names(name, dax)
        lines.append(measure(name, dax, fmt, folder, doc, bool(hidden and hidden[0])))
    lines += ["\tcolumn Value", "\t\tdataType: int64", "\t\tisHidden", f"\t\tlineageTag: {tag('_Measures', 'Value')}",
              "\t\tsummarizeBy: none", "\t\tisNameInferred", "\t\tsourceColumn: [Value]", "",
              "\t\tannotation SummarizationSetBy = Automatic", "",
              "\tpartition _Measures = calculated", "\t\tmode: import", "\t\tsource = { BLANK () }", ""]
    return "\n".join(lines)


PARAMS = [
    ("Miss Cost Multiplier", "GENERATESERIES ( 0.25, 4, 0.25 )", "1", "0.00",
     "multiplies the late/missed cells (200-500) of the cost matrix; 1 = Scania's expert matrix.", "double"),
    ("Workshop Slots", "GENERATESERIES ( 0, 1000, 25 )", "1000", "#,0",
     "call-ins the workshop can take per 1,000 trucks; 1,000 = no limit.", "int64"),
    ("SEK per Cost Unit", "GENERATESERIES ( 250, 5000, 250 )", "1000", "#,0",
     "ILLUSTRATIVE assumption converting relative cost units to SEK (1,000 = an unnecessary call-in of 10 units costs SEK 10,000).", "int64"),
]
RELATIONSHIPS = [(F, "vehicle_id", "dim_vehicle", "vehicle_id"), (F, "recommended_action", "dim_action", "action"),
                 ("shap_top_drivers", "vehicle_id", "dim_vehicle", "vehicle_id")]


def write_model() -> None:
    d = SM / "definition"
    (d / "tables").mkdir(parents=True, exist_ok=True)
    (d / "cultures").mkdir(parents=True, exist_ok=True)
    (SM / ".platform").write_text(json.dumps({
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/gitIntegration/platformProperties/2.0.0/schema.json",
        "metadata": {"type": "SemanticModel", "displayName": NAME},
        "config": {"version": "2.0", "logicalId": tag("semantic-model")}}, indent=2), encoding="utf-8")
    (SM / "definition.pbism").write_text(json.dumps({"version": "4.2", "settings": {}}, indent=2), encoding="utf-8")
    (d / "database.tmdl").write_text("database\n\tcompatibilityLevel: 1606\n\n", encoding="utf-8")
    (d / "cultures" / "en-US.tmdl").write_text("cultureInfo en-US\n\n", encoding="utf-8")
    (d / "expressions.tmdl").write_text(
        f'expression DataFolder = "{DATA_FOLDER}" meta [IsParameterQuery=true, Type="Text", IsParameterQueryRequired=true]\n'
        f"\tlineageTag: {tag('DataFolder')}\n\n\tannotation PBI_ResultType = Text\n\n", encoding="utf-8")
    all_tables = CSV_TABLES + [p[0] for p in PARAMS] + ["Policy", "_Measures"]
    model = ["model Model", "\tculture: en-US", "\tdefaultPowerBIDataSourceVersion: powerBI_V3", "\tsourceQueryCulture: en-US",
             "\tdataAccessOptions", "\t\tlegacyRedirects", "\t\treturnErrorValuesAsNull", "",
             "annotation PBI_ProTooling = [\"DevMode\"]", "",
             "annotation PBI_QueryOrder = " + json.dumps(["DataFolder"] + CSV_TABLES, separators=(",", ":")), ""]
    model += [f"ref table {q(t)}" for t in all_tables] + ["", "ref cultureInfo en-US", ""]
    (d / "model.tmdl").write_text("\n".join(model), encoding="utf-8")
    rel = []
    for ft, fc, tt, tc in RELATIONSHIPS:
        rel += [f"relationship {tag('rel', ft, fc, tt, tc)}", f"\tfromColumn: {ft}.{fc}", f"\ttoColumn: {tt}.{tc}", ""]
    (d / "relationships.tmdl").write_text("\n".join(rel), encoding="utf-8")
    for t in CSV_TABLES:
        (d / "tables" / f"{t}.tmdl").write_text(csv_table(t), encoding="utf-8")
    for p in PARAMS:
        (d / "tables" / f"{p[0]}.tmdl").write_text(parameter_table(*p), encoding="utf-8")
    (d / "tables" / "Policy.tmdl").write_text(policy_table, encoding="utf-8")
    (d / "tables" / "_Measures.tmdl").write_text(measures_table(), encoding="utf-8")


# =============================================================================== report
def lit(v) -> dict:
    return {"expr": {"Literal": {"Value": v}}}


def s(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def col(entity, prop):
    return {"Column": {"Expression": {"SourceRef": {"Entity": entity}}, "Property": prop}}


def meas(prop, entity="_Measures"):
    return {"Measure": {"Expression": {"SourceRef": {"Entity": entity}}, "Property": prop}}


def proj(field, display=None, active=None):
    kind = "Column" if "Column" in field else "Measure"
    ent = field[kind]["Expression"]["SourceRef"]["Entity"]
    prop = field[kind]["Property"]
    p = {"field": field, "queryRef": f"{ent}.{prop}", "nativeQueryRef": prop}
    if active:
        p["active"] = True
    if display:
        p["displayName"] = display
    return p


def titles(title=None, subtitle=None) -> dict:
    out = {}
    if title:
        out["title"] = [{"properties": {"show": lit("true"), "text": lit(s(title)), "titleWrap": lit("true"),
                                        "fontSize": lit("12D"), "bold": lit("true")}}]
    if subtitle:
        out["subTitle"] = [{"properties": {"show": lit("true"), "text": lit(s(subtitle))}}]
    return out


def in_filter(entity, prop, values, name):
    vals = [[{"Literal": {"Value": s(v) if isinstance(v, str) else f"{v}L"}}] for v in values]
    return {"name": name, "field": col(entity, prop), "type": "Categorical",
            "filter": {"Version": 2, "From": [{"Name": "f", "Entity": entity, "Type": 0}],
                       "Where": [{"Condition": {"In": {"Expressions": [
                           {"Column": {"Expression": {"SourceRef": {"Source": "f"}}, "Property": prop}}],
                           "Values": vals}}}]},
            "howCreated": "User"}


class Page:
    def __init__(self, key, display):
        self.name = vid("page", key)
        self.display = display
        self.visuals = []
        self.interactions = []
        self.z = 0

    def add(self, key, x, y, w, h, visual, filters=None):
        name = vid(self.name, key)
        self.z += 1
        v = {"$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/visualContainer/2.12.0/schema.json",
             "name": name, "position": {"x": x, "y": y, "z": self.z * 1000, "height": h, "width": w, "tabOrder": self.z * 1000},
             "visual": visual}
        if filters:
            v["filterConfig"] = {"filters": filters}
        self.visuals.append(v)
        return name

    def no_filter(self, source, target):
        self.interactions.append({"source": source, "target": target, "type": "NoFilter"})

    # ---- visual builders
    def textbox(self, key, x, y, w, h, title, subtitle=None, size="18pt"):
        paras = [{"textRuns": [{"value": title, "textStyle": {"fontSize": size, "fontWeight": "bold", "color": INK}}]}]
        if subtitle:
            paras.append({"textRuns": [{"value": subtitle, "textStyle": {"fontSize": "10pt", "fontWeight": "normal", "color": INK2}}]})
        return self.add(key, x, y, w, h, {"visualType": "textbox", "objects": {"general": [{"properties": {"paragraphs": paras}}]},
                                          "visualContainerObjects": {"background": [{"properties": {"show": lit("false")}}]},
                                          "drillFilterOtherVisuals": True})

    def card(self, key, x, y, w, h, measure_name, display):
        return self.add(key, x, y, w, h, {"visualType": "cardVisual",
                                          "query": {"queryState": {"Data": {"projections": [proj(meas(measure_name), display)]}}},
                                          "drillFilterOtherVisuals": True})

    def slicer(self, key, x, y, w, h, field, display, default=None, single=True):
        objects = {"data": [{"properties": {"mode": lit("'Dropdown'")}}]}
        if single:
            objects["selection"] = [{"properties": {"singleSelect": lit("true")}}]
        if default is not None:
            ent, prop = field["Column"]["Expression"]["SourceRef"]["Entity"], field["Column"]["Property"]
            objects["general"] = [{"properties": {"filter": {"filter": in_filter(ent, prop, [default], "x")["filter"]}}}]
        return self.add(key, x, y, w, h, {"visualType": "slicer",
                                          "query": {"queryState": {"Values": {"projections": [proj(field, display, True)]}}},
                                          "objects": objects, "drillFilterOtherVisuals": True})

    def chart(self, key, x, y, w, h, vtype, category, ys, title, subtitle=None, series=None, colors=None,
              sort=None, filters=None, labels=False, legend=True):
        qs = {"Category": {"projections": [proj(category[0], category[1], True)]},
              "Y": {"projections": [proj(f, d) for f, d in ys]}}
        if series:
            qs["Series"] = {"projections": [proj(series[0], series[1])]}
        query = {"queryState": qs}
        if sort:
            query["sortDefinition"] = {"sort": [{"field": sort[0], "direction": sort[1]}], "isDefaultSort": True}
        objects = {"categoryAxis": [{"properties": {"showAxisTitle": lit("true")}}],
                   "valueAxis": [{"properties": {"showAxisTitle": lit("false")}}],
                   "legend": [{"properties": {"show": lit("true" if legend else "false"),
                                              "position": lit("'TopRight'"), "showTitle": lit("false")}}]}
        if colors:
            objects["dataPoint"] = [{"properties": {"fill": {"solid": {"color": lit(s(c))}}}, "selector": {"metadata": f"_Measures.{m}"}}
                                    for m, c in colors.items()]
        if labels:
            objects["labels"] = [{"properties": {"show": lit("true")}}]
        visual = {"visualType": vtype, "query": query, "objects": objects,
                  "visualContainerObjects": titles(title, subtitle), "drillFilterOtherVisuals": True}
        return self.add(key, x, y, w, h, visual, filters)

    def table(self, key, x, y, w, h, fields, title, subtitle=None, sort=None, filters=None):
        query = {"queryState": {"Values": {"projections": [proj(f, d) for f, d in fields]}}}
        if sort:
            query["sortDefinition"] = {"sort": [{"field": sort[0], "direction": sort[1]}], "isDefaultSort": True}
        return self.add(key, x, y, w, h, {"visualType": "tableEx", "query": query,
                                          "visualContainerObjects": titles(title, subtitle), "drillFilterOtherVisuals": True},
                        filters)

    def matrix(self, key, x, y, w, h, rows, columns, values, title, subtitle=None):
        qs = {"Rows": {"projections": [proj(rows[0], rows[1], True)]},
              "Columns": {"projections": [proj(columns[0], columns[1], True)]},
              "Values": {"projections": [proj(values[0], values[1])]}}
        return self.add(key, x, y, w, h, {"visualType": "pivotTable", "query": {"queryState": qs},
                                          "visualContainerObjects": titles(title, subtitle), "drillFilterOtherVisuals": True})


def build_pages() -> list[Page]:
    pages = []
    W = 1232

    # ------------------------------------------------------------------ 1 Overview
    p = Page("overview", "Overview")
    p.textbox("title", 24, 16, W, 56, "Which trucks to call in for Component X, and what it saves",
              "Test snapshot: 5,045 trucks scored once with the frozen model (notebook 04). Defaults: expert cost matrix, no capacity limit")
    cards = [("Trucks", "Trucks in snapshot"), ("Planner Call-ins per 1000", "Call-ins per 1,000 trucks"),
             ("Planner Share Caught", "Due repairs caught in time"), ("Planner Cost per 1000", "Cost per 1,000 trucks"),
             ("Saving vs Call-everyone SEK per 1000", "Saving vs calling everyone in (SEK, illustrative)")]
    cw = (W - 4 * 16) // 5
    for i, (m, d) in enumerate(cards):
        p.card(f"card{i}", 24 + i * (cw + 16), 88, cw, 100, m, d)
    p.chart("policies", 24, 204, 740, 456, "clusteredBarChart", (col("Policy", "Policy"), "Policy"),
            [(meas("Policy Cost per 1000"), "Cost per 1,000 trucks")],
            "The risk-based plan costs a quarter less than calling every truck in",
            "Cost per 1,000 trucks on the test snapshot, relative units (lower is better)",
            colors={"Policy Cost per 1000": BLUE}, sort=(col("Policy", "Policy"), "Ascending"), labels=True, legend=False)
    p.chart("actions", 780, 204, 476, 456, "clusteredBarChart", (col("dim_action", "label"), "Recommended action"),
            [(meas("Trucks"), "Trucks")], "Recommended actions for today's fleet",
            "Frozen policy (expected-cost rule, expert matrix)", colors={"Trucks": BLUE},
            sort=(col("dim_action", "label"), "Ascending"), labels=True, legend=False)
    p.textbox("footer", 24, 676, W, 32, CITATION, size="8pt")
    pages.append(p)

    # ------------------------------------------------------------------ 2 Service plan
    p = Page("service", "Service plan")
    p.textbox("title", 24, 16, 650, 56, "Service plan: trucks ranked by expected saving",
              "Highest expected saving of calling in first; top 3 risk drivers for trucks the policy acts on")
    p.slicer("s_action", 690, 16, 180, 56, col("dim_action", "label"), "Recommended action", single=False)
    p.slicer("s_spec7", 882, 16, 180, 56, col("dim_vehicle", "Spec_7"), "Spec_7", single=False)
    p.slicer("s_spec1", 1074, 16, 182, 56, col("dim_vehicle", "Spec_1"), "Spec_1", single=False)
    p.table("list", 24, 88, W, 560, [
        (col(F, "priority_rank"), "Priority"), (col(F, "vehicle_id"), "Truck"), (col(F, "age_time_step"), "Age (time steps)"),
        (col(F, "p_due_within_48"), "P(repair due within 48 steps)"), (meas("Planner Action Label"), "Planner action (live)"),
        (col(F, "expected_saving_if_called_in"), "Expected saving if called in"),
        (col(F, "driver_1"), "Driver 1"), (col(F, "driver_1_effect"), "Effect 1"),
        (col(F, "driver_2"), "Driver 2"), (col(F, "driver_3"), "Driver 3")],
        "Today's fleet: who to call in first", "Anonymized variables (e.g. 'counter 666, level'); drivers are SHAP contributions to the risk score",
        sort=(col(F, "priority_rank"), "Ascending"))
    p.textbox("note", 24, 660, W, 40,
              "'Planner action (live)' follows the what-if parameters (defaults here: expert matrix, no capacity limit). "
              "Change them on the Cost trade-off page, or sync those slicers to this page.", size="9pt")
    pages.append(p)

    # ------------------------------------------------------------------ 3 Cost trade-off
    p = Page("tradeoff", "Cost trade-off")
    p.textbox("title", 24, 16, 560, 56, "Cost trade-off: what if breakdowns cost more, or slots are short?",
              "Every truck's action is re-optimised live for the selected values")
    s_m = p.slicer("s_mult", 600, 16, 210, 56, col("Miss Cost Multiplier", "Miss Cost Multiplier"), "Miss-cost multiplier")
    s_k = p.slicer("s_slots", 822, 16, 210, 56, col("Workshop Slots", "Workshop Slots"), "Workshop slots per 1,000 trucks")
    p.slicer("s_sek", 1044, 16, 212, 56, col("SEK per Cost Unit", "SEK per Cost Unit"), "SEK per cost unit (illustrative)")
    cards = [("Planner Cost per 1000", "Cost per 1,000 trucks"), ("Planner Expected Cost per 1000", "Model's own forecast"),
             ("Planner Call-ins per 1000", "Call-ins per 1,000"), ("Planner Share Caught", "Due repairs caught"),
             ("Saving vs Call-everyone SEK per 1000", "Saving vs call-everyone (SEK, illustrative)")]
    for i, (m, d) in enumerate(cards):
        p.card(f"card{i}", 24 + i * (cw + 16), 88, cw, 96, m, d)
    c1 = p.chart("sweep_m", 24, 200, 608, 300, "lineChart", (col("Miss Cost Multiplier", "Miss Cost Multiplier"), "Miss-cost multiplier"),
                 [(meas("Planner Cost per 1000"), "Risk-based plan"), (meas("Call-everyone Cost per 1000"), "Call everyone in"),
                  (meas("Fixed-age Rule Cost per 1000"), "Fixed-age rule (b)")],
                 "The plan beats calling everyone in until a breakdown costs about 2–3× the expert estimate",
                 "Cost per 1,000 trucks vs the miss-cost multiplier (test snapshot, live)",
                 colors={"Planner Cost per 1000": BLUE, "Call-everyone Cost per 1000": ORANGE, "Fixed-age Rule Cost per 1000": AQUA})
    c2 = p.chart("sweep_k", 648, 200, 608, 300, "lineChart", (col("Workshop Slots", "Workshop Slots"), "Workshop slots per 1,000 trucks"),
                 [(meas("Planner Cost per 1000"), "Risk-based plan"), (meas("Call-everyone Cost per 1000"), "Call everyone in")],
                 "A few well-chosen slots deliver most of the saving",
                 "Cost per 1,000 trucks vs workshop capacity; trucks ranked by expected saving (test snapshot, live)",
                 colors={"Planner Cost per 1000": BLUE, "Call-everyone Cost per 1000": ORANGE})
    p.no_filter(s_m, c1)
    p.no_filter(s_k, c2)
    p.chart("valid", 24, 516, W, 180, "lineChart", (col("threshold_sweep", "x"), "Miss-cost multiplier"),
            [(meas("Validation Sweep Cost"), "Cost per 1,000 trucks")],
            "For reference: the same sweep on validation (notebook 04)", "Model re-optimised vs baselines; validation trucks",
            series=(col("threshold_sweep", "policy"), "Policy"),
            filters=[in_filter("threshold_sweep", "sweep", ["miss-cost multiplier"], vid("f", "sweep"))])
    pages.append(p)

    # ------------------------------------------------------------------ 4 Reliability
    p = Page("reliability", "Reliability")
    p.textbox("title", 24, 16, 800, 56, "Reliability: Component X wears out, and some specs fail much faster",
              "Kaplan–Meier on 23,550 training trucks; Weibull and Cox models from notebook 02")
    p.slicer("s_spec", 1040, 16, 216, 56, col("survival_curves", "spec"), "Survival curves for", default="Overall")
    p.chart("km", 24, 88, 760, 420, "lineChart", (col("survival_curves", "time_step"), "Operating time (time steps)"),
            [(meas("Share Without Repair"), "Share without repair")],
            "Share of trucks still on their original Component X", "Pick 'Overall' or a spec; groups under 200 trucks pooled as 'Other'",
            series=(col("survival_curves", "group"), "Group"))
    for i, (m, d) in enumerate([("Weibull Shape", "Weibull shape (> 1 = wear-out)"), ("Repaired by 300", "Repaired by time step 300"),
                                ("Repaired by 400", "Repaired by time step 400")]):
        p.card(f"card{i}", 800 + i * 156, 88, 144, 100, m, d)
    p.table("logrank", 800, 204, 456, 304, [
        (col("logrank_by_spec", "spec"), "Spec"), (col("logrank_by_spec", "chi2"), "χ²"),
        (col("logrank_by_spec", "p_holm_display"), "p (Holm)"), (col("logrank_by_spec", "lowest_group"), "Lowest"),
        (col("logrank_by_spec", "repaired_by_300_lowest"), "Repaired by 300"), (col("logrank_by_spec", "highest_group"), "Highest"),
        (col("logrank_by_spec", "repaired_by_300_highest"), "Repaired by 300 ")],
        "7 of 8 specs differ in survival", "Log-rank test per spec, Holm-corrected", sort=(col("logrank_by_spec", "chi2"), "Descending"))
    p.chart("hr", 24, 524, 760, 172, "clusteredBarChart", (col("cox_hazard_ratios", "label"), "Covariate"),
            [(meas("Hazard Ratio"), "Hazard ratio")], "Heavier early use raises the hazard",
            "Cox model: hazard ratio per 1 SD of log early-life usage rate (1 = no effect)", colors={"Hazard Ratio": BLUE},
            sort=(meas("Hazard Ratio"), "Descending"), labels=True, legend=False,
            filters=[in_filter("cox_hazard_ratios", "kind", ["early-life usage (per 1 SD of log rate)"], vid("f", "hr_usage"))])
    p.table("hr_spec", 800, 524, 456, 172, [
        (col("cox_hazard_ratios", "label"), "Spec category"), (col("cox_hazard_ratios", "hazard_ratio"), "HR"),
        (col("cox_hazard_ratios", "hr_low_95"), "95% low"), (col("cox_hazard_ratios", "hr_high_95"), "95% high"),
        (col("cox_hazard_ratios", "significant_5pct"), "Holm 5%")],
        "Spec categories vs the largest category", None, sort=(col("cox_hazard_ratios", "hazard_ratio"), "Descending"),
        filters=[in_filter("cox_hazard_ratios", "kind", ["spec category (vs largest category)"], vid("f", "hr_spec"))])
    pages.append(p)

    # ------------------------------------------------------------------ 5 Model quality
    p = Page("quality", "Model quality")
    p.textbox("title", 24, 16, W, 56, "Model quality: where the cost comes from, and whether the probabilities can be trusted",
              "Test snapshot, frozen policy; validation numbers from notebooks 03–04")
    p.matrix("confusion", 24, 88, 600, 270, (col(F, "actual_label"), "Actual class"), (col(F, "recommended_label"), "Action"),
             (meas("Trucks"), "Trucks"), "Trucks: actual class vs recommended action", "Below the diagonal = late or missed repairs")
    p.matrix("costs", 640, 88, 616, 270, (col("dim_cost_matrix", "actual_label"), "Actual class"),
             (col("dim_cost_matrix", "predicted_label"), "Action"), (meas("Cost Matrix Cell"), "Cost"),
             "The cost matrix (relative units)", "A missed repair costs 200–500; an unnecessary call-in 7–10")
    p.chart("calib", 24, 374, 600, 322, "clusteredColumnChart", (col("calibration", "bin_label"), "Age band (time steps)"),
            [(meas("Predicted Share Due"), "Predicted"), (meas("Observed Share Due"), "Observed")],
            "Calibrated risk tracks the observed share due across ages", "Test snapshot; P(repair due within 48 steps)",
            colors={"Predicted Share Due": BLUE, "Observed Share Due": GREY},
            filters=[in_filter("calibration", "view", ["age band"], vid("f", "cal_view")),
                     in_filter("calibration", "split", ["test"], vid("f", "cal_split"))])
    p.chart("compare", 640, 374, 616, 322, "clusteredBarChart", (col("policy_comparison", "policy"), "Policy"),
            [(meas("Validation Cost per 1000"), "Validation"), (meas("Test Cost per 1000"), "Test (once)")],
            "The plan held up on unseen trucks", "Cost per 1,000 trucks, notebook 04",
            colors={"Validation Cost per 1000": GREY, "Test Cost per 1000": BLUE},
            sort=(meas("Test Cost per 1000"), "Ascending"))
    pages.append(p)
    return pages


MEASURES.append(("Validation Sweep Cost", "AVERAGE ( threshold_sweep[cost_per_1000] )", "#,0", "Model quality",
                 "Notebook 04 validation sweep: cost per 1,000 trucks at each x."))


def write_report() -> None:
    d = RP / "definition"
    (d / "pages").mkdir(parents=True, exist_ok=True)
    (RP / ".platform").write_text(json.dumps({
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/gitIntegration/platformProperties/2.0.0/schema.json",
        "metadata": {"type": "Report", "displayName": NAME},
        "config": {"version": "2.0", "logicalId": tag("report")}}, indent=2), encoding="utf-8")
    (RP / "definition.pbir").write_text(json.dumps({"version": "4.0", "datasetReference": {"byPath": {"path": f"../{NAME}.SemanticModel"}}},
                                                   indent=2), encoding="utf-8")
    (d / "version.json").write_text(json.dumps({
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/versionMetadata/1.0.0/schema.json",
        "version": "2.0.0"}, indent=2), encoding="utf-8")
    theme_dir = RP / "StaticResources" / "SharedResources" / "BaseThemes"
    theme_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(THEME_SRC, theme_dir / THEME_SRC.name)
    (d / "report.json").write_text(json.dumps({
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/report/3.3.0/schema.json",
        "themeCollection": {"baseTheme": {"name": "Fluent2-CY26SU08",
                                          "reportVersionAtImport": {"visual": "2.12.0", "report": "3.4.0", "page": "2.3.1"},
                                          "type": "SharedResources"}},
        "objects": {"section": [{"properties": {"verticalAlignment": lit("'Top'")}}]},
        "resourcePackages": [{"name": "SharedResources", "type": "SharedResources",
                              "items": [{"name": "Fluent2-CY26SU08", "path": "BaseThemes/Fluent2-CY26SU08.json", "type": "BaseTheme"}]}],
        "settings": {"useStylableVisualContainerHeader": True, "exportDataMode": "AllowSummarized",
                     "defaultDrillFilterOtherVisuals": True, "allowChangeFilterTypes": True,
                     "useEnhancedTooltips": True, "useDefaultAggregateDisplayName": True}}, indent=2), encoding="utf-8")
    pages = build_pages()
    for page in pages:
        pd_ = d / "pages" / page.name
        (pd_ / "visuals").mkdir(parents=True, exist_ok=True)
        meta = {"$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/page/2.1.0/schema.json",
                "name": page.name, "displayName": page.display, "displayOption": "FitToPage", "height": 720, "width": 1280}
        if page.interactions:
            meta["visualInteractions"] = page.interactions
        (pd_ / "page.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        for v in page.visuals:
            (pd_ / "visuals" / v["name"]).mkdir(exist_ok=True)
            (pd_ / "visuals" / v["name"] / "visual.json").write_text(json.dumps(v, indent=2, ensure_ascii=False), encoding="utf-8")
    (d / "pages" / "pages.json").write_text(json.dumps({
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/pagesMetadata/1.1.0/schema.json",
        "pageOrder": [p.name for p in pages], "activePageName": pages[0].name}, indent=2), encoding="utf-8")
    (PBI / f"{NAME}.pbip").write_text(json.dumps({"version": "1.0", "artifacts": [{"report": {"path": f"{NAME}.Report"}}],
                                                  "settings": {"enableAutoRecovery": True}}, indent=2), encoding="utf-8")


def check() -> None:
    """Every field a visual references must exist in the model (column in its table, or a measure)."""
    import re
    tables = {}
    for f in (SM / "definition" / "tables").glob("*.tmdl"):
        txt = f.read_text(encoding="utf-8")
        tname = re.search(r"^table (.+)$", txt, re.M).group(1).strip("'")
        cols_ = {c.strip("'") for c in re.findall(r"^\tcolumn (.+?)(?: =.*)?$", txt, re.M)}
        meas_ = {m.strip("'") for m in re.findall(r"^\tmeasure (.+?) =", txt, re.M)}
        tables[tname] = (cols_, meas_)
    all_measures = {m for _, ms in tables.values() for m in ms}
    n = 0
    for vf in (RP / "definition" / "pages").rglob("visual.json"):
        txt = json.loads(vf.read_text(encoding="utf-8"))
        for m in re.finditer(r'"(Column|Measure)": \{"Expression": \{"SourceRef": \{"Entity": "([^"]+)"\}\}, "Property": "([^"]+)"',
                             json.dumps(txt)):
            kind, ent, prop = m.groups()
            n += 1
            if kind == "Column":
                assert ent in tables and prop in tables[ent][0], f"missing column {ent}[{prop}] in {vf}"
            else:
                assert prop in all_measures, f"missing measure [{prop}] in {vf}"
    print(f"checked {n} field references: all exist in the model")


if __name__ == "__main__":
    # Rebuild the definitions; leave Power BI Desktop's local .pbi cache folders alone.
    for p in (SM / "definition", RP / "definition", RP / "StaticResources"):
        if p.exists():
            shutil.rmtree(p)
    write_model()
    write_report()
    check()
    print(f"wrote {PBI.relative_to(ROOT)}/{NAME}.pbip (DataFolder = {DATA_FOLDER})")
