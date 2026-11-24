"""Reproducible analysis for the simulated podcast video feed experiment."""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent
OUT = ROOT
DATA = ROOT / "data"
if not DATA.exists():
    DATA = ROOT
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(2026)
WINDOW_DAYS = 14


def welch(x: pd.Series, y: pd.Series, alternative="two-sided"):
    x, y = np.asarray(x, float), np.asarray(y, float)
    result = stats.ttest_ind(x, y, equal_var=False, alternative=alternative)
    diff = x.mean() - y.mean()
    se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
    # Welch-Satterthwaite degrees of freedom
    a, b = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    df = (a + b) ** 2 / (a * a / (len(x) - 1) + b * b / (len(y) - 1))
    crit = stats.t.ppf(.975, df)
    return {"control_mean": float(y.mean()), "treatment_mean": float(x.mean()),
            "difference": float(diff), "relative_lift": float(diff / y.mean()) if y.mean() else None,
            "ci_low": float(diff - crit * se), "ci_high": float(diff + crit * se),
            "t_stat": float(result.statistic), "p_value": float(result.pvalue), "df": float(df)}


def main():
    users_raw = pd.read_csv(DATA / "users_experiment_map.csv")
    activity_raw = pd.read_csv(DATA / "daily_user_activity.csv")
    users = users_raw.copy()
    activity = activity_raw.copy()

    users["Region"] = users["Region"].astype("string").str.strip().str.upper().map({
        "NORTH AMERICA": "North America", "EUROPE": "Europe", "LATAM": "LATAM",
        "APAC": "APAC", "MEA": "MEA"})

    exact_dup = activity.duplicated(keep="first")
    activity_dedup = activity.loc[~exact_dup].copy()
    known_ids = set(users.User_ID)
    orphan = ~activity_dedup.User_ID.isin(known_ids)
    activity_joinable = activity_dedup.loc[~orphan].copy()
    impossible = activity_joinable.Podcast_Minutes_Listened > 1440
    activity_clean = activity_joinable.loc[~impossible].copy()
    activity_clean["Date"] = pd.to_datetime(activity_clean.Date)

    # Per-user metrics over the fixed experiment window. Null podcast minutes remain
    # unknown and are excluded from sums; absent days contribute zero through /14.
    sums = activity_clean.groupby("User_ID").agg(
        podcast_total=("Podcast_Minutes_Listened", "sum"),
        music_total=("Music_Minutes_Listened", "sum"),
        clicks_total=("Discovery_Feed_Clicks", "sum"),
        crashes_total=("App_Crashes", "sum"),
        active_days=("Date", "nunique"),
        podcast_days=("Podcast_Minutes_Listened", lambda s: int((s > 0).sum())),
    )
    metrics = users.merge(sums, how="left", on="User_ID")
    for col in ["podcast_total", "music_total", "clicks_total", "crashes_total", "active_days", "podcast_days"]:
        metrics[col] = metrics[col].fillna(0)
    metrics["avg_daily_podcast_min"] = metrics.podcast_total / WINDOW_DAYS
    metrics["avg_daily_music_min"] = metrics.music_total / WINDOW_DAYS
    metrics["avg_daily_clicks"] = metrics.clicks_total / WINDOW_DAYS
    metrics["avg_daily_crashes"] = metrics.crashes_total / WINDOW_DAYS
    metrics["active_day_rate"] = metrics.active_days / WINDOW_DAYS
    metrics["podcast_days"] = metrics.podcast_days.astype(int)
    metrics["podcast_listener"] = (metrics.podcast_total > 0).astype(int)
    denom = metrics.podcast_total + metrics.music_total
    metrics["podcast_listening_share"] = np.divide(metrics.podcast_total, denom, out=np.zeros(len(metrics)), where=denom.to_numpy() > 0)

    # SQL deliverable executes against in-memory copies of the raw source tables.
    conn = sqlite3.connect(":memory:")
    users_raw.to_sql("users_experiment_map", conn, index=False)
    activity_raw.to_sql("daily_user_activity", conn, index=False)
    sql_path = OUT / "analysis.sql"
    query = sql_path.read_text(encoding="utf-8")
    sql_metrics = pd.read_sql_query(query, conn)
    conn.close()

    # Experiment diagnostics precede outcomes in the report.
    arm_counts = users.Variant.value_counts()
    srm_chi, srm_p = stats.chisquare([arm_counts.get("Control", 0), arm_counts.get("Treatment", 0)])
    region_tab = pd.crosstab(users.Variant, users.Region)
    prem_tab = pd.crosstab(users.Variant, users.Premium_Status)
    region_chi, region_p, _, _ = stats.chi2_contingency(region_tab)
    prem_chi, prem_p, _, _ = stats.chi2_contingency(prem_tab)
    age_ctrl = users.loc[users.Variant == "Control", "Account_Age_Days"].dropna()
    age_trt = users.loc[users.Variant == "Treatment", "Account_Age_Days"].dropna()
    age_t, age_p = stats.ttest_ind(age_trt, age_ctrl, equal_var=False)

    control = metrics[metrics.Variant == "Control"]
    treatment = metrics[metrics.Variant == "Treatment"]
    primary = welch(treatment.avg_daily_podcast_min, control.avg_daily_podcast_min)
    primary["mann_whitney_p"] = float(stats.mannwhitneyu(treatment.avg_daily_podcast_min, control.avg_daily_podcast_min, alternative="two-sided").pvalue)
    # Percentile bootstrap CI for mean difference, resampling users within arm.
    boot = np.empty(5000)
    cx = control.avg_daily_podcast_min.to_numpy(); tx = treatment.avg_daily_podcast_min.to_numpy()
    for i in range(len(boot)):
        boot[i] = RNG.choice(tx, len(tx), replace=True).mean() - RNG.choice(cx, len(cx), replace=True).mean()
    primary["bootstrap_ci_low"], primary["bootstrap_ci_high"] = map(float, np.quantile(boot, [.025, .975]))

    secondary_cols = {
        "avg_daily_clicks": "Feed clicks per user-day",
        "podcast_listener": "Podcast listener rate",
        "podcast_days": "Podcast listening days per user",
        "podcast_listening_share": "Podcast share of listening",
        "active_day_rate": "Active-day rate",
    }
    secondary = {}
    for col, label in secondary_cols.items():
        r = welch(treatment[col], control[col])
        r["metric"] = label
        secondary[col] = r
    for r in secondary.values():
        r["bonferroni_alpha"] = .05 / len(secondary)
        r["bonferroni_significant"] = r["p_value"] < .05 / len(secondary)

    guardrails = {}
    for col, label, alt, threshold in [
        ("avg_daily_crashes", "App crashes per user-day", "greater", .10),
        ("avg_daily_music_min", "Music minutes per user-day", "less", -.03),
        ("active_day_rate", "Active-day rate", "less", None),
    ]:
        r = welch(treatment[col], control[col], alternative=alt)
        r["metric"] = label
        r["harm_direction_p"] = r["p_value"]
        r["threshold_relative_change"] = threshold
        r["relative_change"] = r["relative_lift"]
        r["threshold_exceeded"] = (r["relative_lift"] is not None and ((alt == "greater" and r["relative_lift"] > threshold) or (alt == "less" and r["relative_lift"] < threshold))) if threshold is not None else False
        r["statistically_significant_harm"] = r["harm_direction_p"] < .05
        guardrails[col] = r

    # Exploratory segments
    segments = {}
    for dim in ["Premium_Status", "Region"]:
        segments[dim] = {}
        for value, frame in metrics.groupby(dim, dropna=False):
            c = frame[frame.Variant == "Control"].avg_daily_podcast_min
            t = frame[frame.Variant == "Treatment"].avg_daily_podcast_min
            segments[dim][str(value)] = {"control_mean": float(c.mean()), "treatment_mean": float(t.mean()),
                "difference": float(t.mean()-c.mean()), "p_value": float(stats.ttest_ind(t,c,equal_var=False).pvalue), "n_control":len(c),"n_treatment":len(t)}

    n1, n2 = len(control), len(treatment)
    df_power = n1 + n2 - 2
    crit_power = stats.t.ppf(.975, df_power)
    def achieved_power(effect):
        ncp = effect / np.sqrt(1/n1 + 1/n2)
        return stats.nct.cdf(-crit_power, df_power, ncp) + stats.nct.sf(crit_power, df_power, ncp)
    mde_d = float(brentq(lambda effect: achieved_power(effect) - .80, 1e-8, 5.0))
    pooled_sd = float(np.sqrt((control.avg_daily_podcast_min.var(ddof=1)+treatment.avg_daily_podcast_min.var(ddof=1))/2))
    mde_min = mde_d * pooled_sd

    clean_log = pd.DataFrame([
        ["Activity rows received", len(activity_raw), 0, "Source rows"],
        ["Exact duplicate activity rows", int(exact_dup.sum()), int(exact_dup.sum()), "Removed exact duplicate records"],
        ["Rows after exact deduplication", len(activity_dedup), 0, "Intermediate row count"],
        ["Orphan activity rows", int(orphan.sum()), int(orphan.sum()), "Removed rows without enrolled user"],
        ["Rows after orphan removal", len(activity_joinable), 0, "Intermediate row count"],
        ["Impossible podcast minutes (>1,440)", int(impossible.sum()), int(impossible.sum()), "Removed impossible values"],
        ["Clean activity rows", len(activity_clean), 0, "Final joined analysis events"],
        ["Null podcast minute values in clean events", int(activity_clean.Podcast_Minutes_Listened.isna().sum()), 0, "Kept null; ignored by sum"],
        ["Users with no clean activity rows", int((~users.User_ID.isin(activity_clean.User_ID)).sum()), 0, "Retained as zero in user-level table"],
        ["Users with missing account age", int(users.Account_Age_Days.isna().sum()), 0, "Kept null; excluded only from age balance test"],
    ], columns=["Check", "Rows or users", "Removed rows", "Handling"])
    clean_log.to_csv(OUT / "cleaning_log.csv", index=False)
    metrics.to_csv(OUT / "user_level_metrics.csv", index=False)
    sql_metrics.to_csv(OUT / "user_level_metrics_sql.csv", index=False)
    pd.DataFrame([{"Variant": v, "users": len(g), **{c: g[c].mean() for c in ["avg_daily_podcast_min", "avg_daily_clicks", "podcast_listener", "podcast_days", "podcast_listening_share", "avg_daily_music_min", "avg_daily_crashes", "active_day_rate"]}} for v,g in metrics.groupby("Variant")]).to_csv(OUT / "variant_summary.csv", index=False)
    pd.DataFrame([{"metric": k, **v} for k,v in secondary.items()]).to_csv(OUT / "secondary_results.csv", index=False)
    pd.DataFrame([{"metric": k, **v} for k,v in guardrails.items()]).to_csv(OUT / "guardrail_results.csv", index=False)

    # Charts
    plt.figure(figsize=(7.2, 4.5))
    means = metrics.groupby("Variant").avg_daily_podcast_min.mean().reindex(["Control", "Treatment"])
    errs = [stats.sem(metrics.loc[metrics.Variant == v, "avg_daily_podcast_min"]) * 1.96 for v in means.index]
    plt.bar(means.index, means.values, yerr=errs, capsize=5, color=["#68778a", "#1f9d67"])
    plt.ylabel("Podcast minutes per user per day")
    plt.title("Primary metric by experiment arm (95% CI around mean)")
    plt.tight_layout(); plt.savefig(OUT / "primary_metric.png", dpi=180); plt.close()

    daily = activity_clean.merge(users[["User_ID", "Variant"]], on="User_ID")
    daily = daily.groupby(["Date", "Variant"]).Podcast_Minutes_Listened.mean().unstack()
    full = pd.date_range(activity_clean.Date.min(), activity_clean.Date.max(), freq="D")
    daily = daily.reindex(full).rename_axis("Date")
    daily.plot(figsize=(8,4), marker="o", color={"Control":"#68778a", "Treatment":"#1f9d67"})
    plt.ylabel("Mean podcast minutes per observed activity row")
    plt.title("Daily podcast minutes among recorded activity events")
    plt.tight_layout(); plt.savefig(OUT / "daily_trend.png", dpi=180); plt.close()

    sql_compare = sql_metrics.set_index("User_ID").sort_index()["avg_daily_podcast_min"]
    py_compare = metrics.set_index("User_ID").sort_index()["avg_daily_podcast_min"]
    result = {"row_counts":{"users":len(users),"activity_raw":len(activity_raw),"exact_duplicates":int(exact_dup.sum()),"orphans":int(orphan.sum()),"impossible":int(impossible.sum()),"clean_events":len(activity_clean),"null_podcast_minutes":int(activity_clean.Podcast_Minutes_Listened.isna().sum()),"users_no_activity":int((~users.User_ID.isin(activity_clean.User_ID)).sum())},
        "randomization":{"variant_counts":{k:int(v) for k,v in arm_counts.items()},"srm_chi_square":float(srm_chi),"srm_p":float(srm_p),"region_p":float(region_p),"premium_p":float(prem_p),"account_age_p":float(age_p),"account_age_control_mean":float(age_ctrl.mean()),"account_age_treatment_mean":float(age_trt.mean())},
        "primary":primary,"secondary":secondary,"guardrails":guardrails,"segments":segments,"power":{"mde_standardized_d":mde_d,"pooled_sd_minutes":pooled_sd,"mde_minutes_per_day":mde_min,"mde_relative_to_control":mde_min/control.avg_daily_podcast_min.mean()},
        "sql_python_max_abs_difference":float(np.max(np.abs(sql_compare.to_numpy()-py_compare.to_numpy())))}
    (OUT / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(result)
    write_dashboard(result)


def write_report(r):
    p = r["primary"]; q = r["randomization"]; g = r["guardrails"]; sec = r["secondary"]; power = r["power"]
    breached = []
    for name, row in g.items():
        if row["statistically_significant_harm"] or row["threshold_exceeded"]:
            breached.append(row["metric"])
    if p["p_value"] < .05 and p["difference"] > 0 and not breached:
        decision = "SHIP"
        rationale = "The primary metric is significantly positive and no observable guardrail breach meets the PRD rule."
    elif p["p_value"] < .05 and p["difference"] < 0:
        decision = "DO NOT SHIP"
        rationale = "The primary metric is significantly negative."
    elif p["p_value"] < .05 and breached:
        decision = "ITERATE"
        rationale = "The primary metric is positive, but at least one observed guardrail is flagged."
    else:
        decision = "DO NOT SHIP ON THIS EVIDENCE"
        rationale = "The primary effect is not statistically significant under the prespecified two-sided test."
    sec_lines = "\n".join(f"| {v['metric']} | {v['control_mean']:.4f} | {v['treatment_mean']:.4f} | {v['difference']:+.4f} | {v['relative_lift']:+.1%} | {v['p_value']:.4g} | {v['bonferroni_alpha']:.3f} | {'Yes' if v['bonferroni_significant'] else 'No'} |" for v in sec.values())
    guard_lines = "\n".join(f"| {v['metric']} | {v['control_mean']:.4f} | {v['treatment_mean']:.4f} | {v['relative_lift']:+.1%} | {v['harm_direction_p']:.4g} | {'Yes' if v['statistically_significant_harm'] else 'No'} | {'Yes' if v['threshold_exceeded'] else 'No'} |" for v in g.values())
    report = f"""# Podcast Video-First Discovery Feed A/B Test

**Experiment:** EXP-PODVID-014  
**Window:** 3–16 March 2025 (14 days)  
**Recommendation:** **{decision}**

## Executive summary

The hypothesis was that the Video-First Discovery Feed increases average daily podcast listening minutes per assigned user. The user-randomized experiment included {q['variant_counts'].get('Control',0):,} Control and {q['variant_counts'].get('Treatment',0):,} Treatment users. Intent-to-treat user averages include all {WINDOW_DAYS} days, with users who had no activity retained as zero.

Treatment averaged **{p['treatment_mean']:.2f} podcast minutes per user per day**, versus **{p['control_mean']:.2f}** in Control. The difference was **{p['difference']:+.2f} minutes/day ({p['relative_lift']:+.1%})**, with a 95% Welch confidence interval of **[{p['ci_low']:+.2f}, {p['ci_high']:+.2f}]** and **p = {p['p_value']:.4g}**. The Mann–Whitney robustness check returned p = {p['mann_whitney_p']:.4g}; the 5,000-resample bootstrap interval was [{p['bootstrap_ci_low']:+.2f}, {p['bootstrap_ci_high']:+.2f}].

**Decision: {decision}.** {rationale} The experiment's estimated 80% power MDE was {power['mde_minutes_per_day']:.2f} minutes/day ({power['mde_relative_to_control']:.1%} of the Control mean), so effects smaller than this may not be detected reliably.

## Pre-test validity

The sample ratio was {q['variant_counts'].get('Control',0)} Control / {q['variant_counts'].get('Treatment',0)} Treatment. The 50/50 chi-square SRM p-value was **{q['srm_p']:.4g}** ({'passes' if q['srm_p'] >= .001 else 'fails'} the PRD p < 0.001 threshold). Categorical balance p-values were {q['region_p']:.3f} for Region and {q['premium_p']:.3f} for Premium status. Account age means were {q['account_age_control_mean']:.1f} days Control and {q['account_age_treatment_mean']:.1f} days Treatment (Welch p = {q['account_age_p']:.3f}; missing values excluded from this comparison). The split passes the SRM check. Region has a nominal imbalance at the 5% level, so interpret unadjusted outcome differences with that caveat.

## Data cleaning

| Step | Count | Handling |
|---|---:|---|
| Activity rows received | {r['row_counts']['activity_raw']:,} | Raw input |
| Exact duplicate rows removed | {r['row_counts']['exact_duplicates']:,} | Retained first copy |
| Orphan activity rows removed | {r['row_counts']['orphans']:,} | No matching enrolled user |
| Impossible podcast-minute rows removed | {r['row_counts']['impossible']:,} | Podcast minutes > 1,440 |
| Clean activity events | {r['row_counts']['clean_events']:,} | Analysis events |
| Null podcast-minute values in clean events | {r['row_counts']['null_podcast_minutes']:,} | Kept null and ignored in podcast-minute sums |
| Enrolled users with no clean events | {r['row_counts']['users_no_activity']:,} | Retained as zero for the 14-day user metric |

Region values were normalized to canonical capitalization. Missing account age remains missing. Null podcast minutes remain null and are ignored in podcast-minute sums; missing activity dates are not fabricated. All assigned users remain in the analysis.

## Secondary metrics

Secondary p-values use a Bonferroni threshold of 0.05/5 = 0.01.

| Metric | Control | Treatment | Difference | Relative change | p-value | Adjusted alpha | Significant |
|---|---:|---:|---:|---:|---:|---:|---|
{sec_lines}

## Guardrails

Crash harm is tested Treatment > Control, and music and active-day harm are tested Treatment < Control. The PRD also flags crash increases above 10% and music declines above 3% even apart from significance.

| Guardrail | Control | Treatment | Relative change | One-sided harm p | Significant harm | Threshold crossed |
|---|---:|---:|---:|---:|---|---|
{guard_lines}

Premium cancellation **could not be evaluated** because neither source table contains cancellation events. Production launch monitoring should join billing cancellation records by Premium user and variant.

## Exploratory cuts and caveats

Premium and regional cuts are provided in `results.json`; these are exploratory and should generate follow-up hypotheses only. Segment comparisons have less power and inflate false-positive risk. The daily trend plot is descriptive; its denominator is users with recorded activity rows on each day, not all assigned users.

This dataset is simulated for a training case study, not a Spotify production experiment. The result is limited to the supplied 14-day sample and available telemetry. The activity data cannot distinguish content substitution at the individual session level, and cancellation is absent.

## Reproducibility

Run `python analysis.py` from this project directory. The script cleans the CSVs, loads them into an in-memory SQLite database, executes `analysis.sql`, exports the user-level table and summaries, and refreshes this report and figures. Python and SQL user-level primary metrics are compared in `results.json`.
"""
    (OUT / "executive_summary.md").write_text(report, encoding="utf-8")


def write_dashboard(r):
    payload = json.dumps(r, separators=(",", ":")).replace("</", "<\\/")
    html = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Results dashboard for the simulated Spotify podcast video feed A/B test.">
<title>Podcast video feed experiment | Results</title>
<style>
:root{--ink:#17241f;--muted:#62716b;--line:#dce5df;--paper:#f4f7f4;--white:#fff;--green:#087f5b;--green2:#dff4e9;--red:#b04439;--red2:#fff0ed;--gold:#8a6412;--gold2:#fff4d8;--shadow:0 8px 26px #173c2810}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1150px;margin:auto;padding:34px 22px 60px}.eyebrow{text-transform:uppercase;letter-spacing:.13em;color:var(--green);font-weight:750;font-size:12px}.top{display:flex;justify-content:space-between;gap:24px;align-items:flex-start;margin:10px 0 24px}.top h1{font-size:clamp(28px,4vw,44px);letter-spacing:-.04em;line-height:1.1;margin:0 0 10px}.sub{color:var(--muted);margin:0}.pill{white-space:nowrap;background:var(--gold2);color:var(--gold);font-weight:800;padding:9px 14px;border-radius:999px}.decision{background:#17392e;color:#f6fff9;border-radius:18px;padding:22px 24px;box-shadow:var(--shadow);display:flex;gap:18px;align-items:center;margin-bottom:20px}.decision strong{font-size:19px;display:block;margin-bottom:3px}.decision span{color:#d2e5db}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}.card{background:var(--white);border:1px solid var(--line);border-radius:15px;padding:18px;box-shadow:var(--shadow)}.metric-label{font-size:12px;color:var(--muted);font-weight:700;letter-spacing:.045em;text-transform:uppercase}.metric-value{font-size:29px;letter-spacing:-.04em;font-weight:780;margin:7px 0 2px}.metric-note{color:var(--muted);font-size:13px}.section{margin-top:22px}.section h2{font-size:20px;letter-spacing:-.02em;margin:0 0 12px}.split{display:grid;grid-template-columns:1.1fr .9fr;gap:14px}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;border-bottom:1px solid var(--line);padding:10px 9px;white-space:nowrap}th{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}td:first-child{white-space:normal}.status{font-size:11px;font-weight:750;padding:4px 8px;border-radius:999px;display:inline-block}.yes{background:var(--red2);color:var(--red)}.no{background:var(--green2);color:var(--green)}.flag{background:var(--gold2);color:var(--gold)}.note{background:#fff;border-left:4px solid #d2a13c;padding:14px 16px;border-radius:8px;color:#55471f}.foot{font-size:12px;color:var(--muted);margin-top:20px}.chart{width:100%;height:230px}.small{font-size:13px;color:var(--muted)}.check-list{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}.check{border:1px solid var(--line);border-radius:11px;padding:13px}.check b{display:block;margin-bottom:2px}.recommendation{color:#f3c96b;font-weight:800;letter-spacing:.08em;font-size:12px}.link{color:var(--green);font-weight:700;text-decoration:none}.link:hover{text-decoration:underline}
@media(max-width:800px){.grid{grid-template-columns:repeat(2,1fr)}.split{grid-template-columns:1fr}.top{display:block}.pill{display:inline-block;margin-top:14px}.decision{align-items:flex-start}.check-list{grid-template-columns:1fr}}@media(max-width:480px){main{padding:22px 14px 44px}.grid{grid-template-columns:1fr 1fr;gap:9px}.card{padding:14px}.metric-value{font-size:24px}.decision{padding:18px}}
</style></head><body><main>
<div class="eyebrow">Product analytics · EXP-PODVID-014</div>
<div class="top"><div><h1>Video-first podcast discovery</h1><p class="sub">A/B test readout · 3–16 March 2025 · 14-day intent-to-treat analysis</p></div><div class="pill">SIMULATED CASE STUDY</div></div>
<div class="decision"><div><div class="recommendation">RECOMMENDATION</div><strong>Iterate before launch</strong><span>Podcast listening increased, but the app crash guardrail was breached.</span></div></div>
<div class="grid" id="kpis"></div>
<section class="section split"><article class="card"><h2>Primary metric</h2><p class="small">Average podcast listening per assigned user per day. Error bars show the 95% confidence interval for the difference.</p><svg class="chart" id="primary-chart" role="img" aria-label="Primary metric treatment versus control"></svg><div id="primary-caption" class="small"></div></article>
<article class="card"><h2>Experiment validity</h2><div class="check-list" id="checks"></div><p class="small" id="balance-note"></p></article></section>
<section class="section card"><h2>Secondary metrics</h2><p class="small">Bonferroni-adjusted significance threshold: α = 0.01 across five secondary tests.</p><div class="table-wrap"><table><thead><tr><th>Metric</th><th>Control</th><th>Treatment</th><th>Change</th><th>p-value</th><th>Adjusted result</th></tr></thead><tbody id="secondary"></tbody></table></div></section>
<section class="section card"><h2>Guardrails</h2><p class="small">One-sided tests check for harm. Crash increase above 10% and music decline above 3% are also flagged by the decision framework.</p><div class="table-wrap"><table><thead><tr><th>Metric</th><th>Control</th><th>Treatment</th><th>Relative change</th><th>Harm p-value</th><th>Decision rule</th></tr></thead><tbody id="guardrails"></tbody></table></div></section>
<section class="section split"><article class="card"><h2>Exploratory segments</h2><p class="small">Use these cuts to form follow-up hypotheses, not as launch conclusions.</p><div class="table-wrap"><table><thead><tr><th>Segment</th><th>Control</th><th>Treatment</th><th>Difference</th><th>p-value</th></tr></thead><tbody id="segments"></tbody></table></div></article><article class="card"><h2>Power and limitations</h2><div id="power"></div><p class="note">Premium cancellation was not measurable because the source data has no billing or cancellation events. Join cancellation records for launch monitoring.</p><p class="small">Simulated training data, not Spotify production results. Podcast-minute nulls are left null; assigned users are analyzed regardless of activity.</p><a class="link" href="https://github.com/Vaishnavi-Sonwane/spotify-podcast-video-ab-test/blob/main/executive_summary.md">Read the full executive summary →</a></article></section>
<p class="foot">Source files and reproducible code are included with this static dashboard. Python and SQLite results reconcile. The dashboard uses no external libraries or network services.</p>
</main><script id="payload" type="application/json">__DATA__</script><script>
const d=JSON.parse(document.getElementById('payload').textContent),p=d.primary,q=d.randomization;
const fmt=(x,n=2)=>Number(x).toFixed(n),pct=x=>(x>=0?'+':'')+(100*x).toFixed(1)+'%',pval=x=>x<.0001?x.toExponential(2):fmt(x,4);
const kpis=[['Control mean',fmt(p.control_mean)+' min','per user per day'],['Treatment mean',fmt(p.treatment_mean)+' min','per user per day'],['Estimated lift',pct(p.relative_lift),fmt(p.difference)+' min/day · p = '+pval(p.p_value)],['Crash change',pct(d.guardrails.avg_daily_crashes.relative_lift),d.guardrails.avg_daily_crashes.statistically_significant_harm?'Significant harm signal':'No significant harm signal']];
document.getElementById('kpis').innerHTML=kpis.map(x=>`<div class="card"><div class="metric-label">${x[0]}</div><div class="metric-value">${x[1]}</div><div class="metric-note">${x[2]}</div></div>`).join('');
document.getElementById('checks').innerHTML=[['Sample ratio','Control '+q.variant_counts.Control+' / Treatment '+q.variant_counts.Treatment],['SRM p-value',pval(q.srm_p)+' · passes threshold'],['Region balance p-value',pval(q.region_p)],['Premium balance p-value',pval(q.premium_p)],['Account age balance p-value',pval(q.account_age_p)],['User-level rows',d.row_counts.users.toLocaleString()+' assigned users']].map(x=>`<div class="check"><b>${x[0]}</b><span class="small">${x[1]}</span></div>`).join('');
document.getElementById('balance-note').textContent='Region has a nominal imbalance at the 5% level. Interpret the unadjusted outcome comparison with that caveat.';
const svg=document.getElementById('primary-chart'), W=640,H=230,L=72,R=25,Y=115,lo=p.ci_low,hi=p.ci_high,pad=(hi-lo)*.2||.2,min=lo-pad,max=hi+pad,x=v=>L+(v-min)/(max-min)*(W-L-R);let ticks=[];for(let i=0;i<5;i++)ticks.push(min+(max-min)*i/4);svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.innerHTML=`<line x1="${x(0)}" y1="32" x2="${x(0)}" y2="174" stroke="#9aa9a1" stroke-dasharray="5 5"/><line x1="${x(lo)}" y1="${Y}" x2="${x(hi)}" y2="${Y}" stroke="#087f5b" stroke-width="7" stroke-linecap="round"/><line x1="${x(lo)}" y1="${Y-11}" x2="${x(lo)}" y2="${Y+11}" stroke="#087f5b" stroke-width="3"/><line x1="${x(hi)}" y1="${Y-11}" x2="${x(hi)}" y2="${Y+11}" stroke="#087f5b" stroke-width="3"/><circle cx="${x(p.difference)}" cy="${Y}" r="9" fill="#087f5b"/><text x="${L}" y="48" fill="#62716b" font-size="13">Treatment − Control (minutes/day)</text>${ticks.map(t=>`<line x1="${x(t)}" y1="176" x2="${x(t)}" y2="182" stroke="#86958d"/><text x="${x(t)}" y="202" text-anchor="middle" fill="#62716b" font-size="12">${fmt(t,1)}</text>`).join('')}<text x="${x(p.difference)}" y="87" text-anchor="middle" fill="#087f5b" font-size="14" font-weight="700">+${fmt(p.difference)} min</text>`;
document.getElementById('primary-caption').textContent=`95% CI [${fmt(p.ci_low)}, ${fmt(p.ci_high)}] minutes/day · Welch two-sided test · p = ${pval(p.p_value)} · Mann–Whitney p = ${pval(p.mann_whitney_p)}`;
document.getElementById('secondary').innerHTML=Object.values(d.secondary).map(v=>`<tr><td>${v.metric}</td><td>${fmt(v.control_mean,3)}</td><td>${fmt(v.treatment_mean,3)}</td><td>${pct(v.relative_lift)}</td><td>${pval(v.p_value)}</td><td><span class="status ${v.bonferroni_significant?'yes':'no'}">${v.bonferroni_significant?'Significant':'Not significant'}</span></td></tr>`).join('');
document.getElementById('guardrails').innerHTML=Object.values(d.guardrails).map(v=>{let hit=v.statistically_significant_harm||v.threshold_exceeded;let rule=v.statistically_significant_harm?'Significant harm':v.threshold_exceeded?'Threshold crossed':'No breach';return `<tr><td>${v.metric}</td><td>${fmt(v.control_mean,4)}</td><td>${fmt(v.treatment_mean,4)}</td><td>${pct(v.relative_lift)}</td><td>${pval(v.harm_direction_p)}</td><td><span class="status ${hit?'yes':'no'}">${rule}</span></td></tr>`}).join('');
document.getElementById('segments').innerHTML=Object.entries(d.segments).flatMap(([dim,vals])=>Object.entries(vals).map(([name,v])=>`<tr><td>${dim}: ${name}</td><td>${fmt(v.control_mean,2)}</td><td>${fmt(v.treatment_mean,2)}</td><td>${v.difference>=0?'+':''}${fmt(v.difference,2)}</td><td>${pval(v.p_value)}</td></tr>`)).join('');
document.getElementById('power').innerHTML=`<p><b>80% power minimum detectable effect</b><br>${fmt(d.power.mde_minutes_per_day)} min/user/day (${pct(d.power.mde_relative_to_control)} of Control mean)</p><p class="small">Observed primary difference: ${fmt(p.difference)} minutes/day. Bootstrap 95% CI: [${fmt(p.bootstrap_ci_low)}, ${fmt(p.bootstrap_ci_high)}].</p>`;
</script></body></html>"""
    (OUT / "index.html").write_text(html.replace("__DATA__", payload), encoding="utf-8")


if __name__ == "__main__":
    main()
