# Podcast Video-First Discovery Feed A/B Test

**Experiment:** EXP-PODVID-014  
**Window:** 3–16 March 2025 (14 days)  
**Recommendation:** **ITERATE**

## Executive summary

The hypothesis was that the Video-First Discovery Feed increases average daily podcast listening minutes per assigned user. The user-randomized experiment included 1,015 Control and 985 Treatment users. Intent-to-treat user averages include all 14 days, with users who had no activity retained as zero.

Treatment averaged **11.42 podcast minutes per user per day**, versus **10.35** in Control. The difference was **+1.08 minutes/day (+10.4%)**, with a 95% Welch confidence interval of **[+0.21, +1.94]** and **p = 0.01489**. The Mann–Whitney robustness check returned p = 0.0578; the 5,000-resample bootstrap interval was [+0.21, +1.97].

**Decision: ITERATE.** The primary metric is positive, but at least one observed guardrail is flagged. The experiment's estimated 80% power MDE was 1.24 minutes/day (11.9% of the Control mean), so effects smaller than this may not be detected reliably.

## Pre-test validity

The sample ratio was 1015 Control / 985 Treatment. The 50/50 chi-square SRM p-value was **0.5023** (passes the PRD p < 0.001 threshold). Categorical balance p-values were 0.026 for Region and 0.947 for Premium status. Account age means were 845.3 days Control and 807.4 days Treatment (Welch p = 0.138; missing values excluded from this comparison). The split passes the SRM check. Region has a nominal imbalance at the 5% level, so interpret unadjusted outcome differences with that caveat.

## Data cleaning

| Step | Count | Handling |
|---|---:|---|
| Activity rows received | 19,741 | Raw input |
| Exact duplicate rows removed | 78 | Retained first copy |
| Orphan activity rows removed | 20 | No matching enrolled user |
| Impossible podcast-minute rows removed | 12 | Podcast minutes > 1,440 |
| Clean activity events | 19,631 | Analysis events |
| Null podcast-minute values in clean events | 157 | Kept null and ignored in podcast-minute sums |
| Enrolled users with no clean events | 0 | Retained as zero for the 14-day user metric |

Region values were normalized to canonical capitalization. Missing account age remains missing. Null podcast minutes remain null and are ignored in podcast-minute sums; missing activity dates are not fabricated. All assigned users remain in the analysis.

## Secondary metrics

Secondary p-values use a Bonferroni threshold of 0.05/5 = 0.01.

| Metric | Control | Treatment | Difference | Relative change | p-value | Adjusted alpha | Significant |
|---|---:|---:|---:|---:|---:|---:|---|
| Feed clicks per user-day | 0.6881 | 0.9860 | +0.2979 | +43.3% | 4.31e-21 | 0.010 | Yes |
| Podcast listener rate | 0.9970 | 0.9959 | -0.0011 | -0.1% | 0.6765 | 0.010 | No |
| Podcast listening days per user | 6.7222 | 6.9939 | +0.2717 | +4.0% | 0.008979 | 0.010 | Yes |
| Podcast share of listening | 0.2063 | 0.2164 | +0.0101 | +4.9% | 0.07958 | 0.010 | No |
| Active-day rate | 0.6950 | 0.7074 | +0.0124 | +1.8% | 0.1339 | 0.010 | No |

## Guardrails

Crash harm is tested Treatment > Control, and music and active-day harm are tested Treatment < Control. The PRD also flags crash increases above 10% and music declines above 3% even apart from significance.

| Guardrail | Control | Treatment | Relative change | One-sided harm p | Significant harm | Threshold crossed |
|---|---:|---:|---:|---:|---|---|
| App crashes per user-day | 0.0161 | 0.0191 | +18.8% | 0.03181 | Yes | Yes |
| Music minutes per user-day | 38.5425 | 38.4733 | -0.2% | 0.4619 | No | No |
| Active-day rate | 0.6950 | 0.7074 | +1.8% | 0.933 | No | No |

Premium cancellation **could not be evaluated** because neither source table contains cancellation events. Production launch monitoring should join billing cancellation records by Premium user and variant.

## Exploratory cuts and caveats

Premium and regional cuts are provided in `results.json`; these are exploratory and should generate follow-up hypotheses only. Segment comparisons have less power and inflate false-positive risk. The daily trend plot is descriptive; its denominator is users with recorded activity rows on each day, not all assigned users.

This dataset is simulated for a training case study, not a Spotify production experiment. The result is limited to the supplied 14-day sample and available telemetry. The activity data cannot distinguish content substitution at the individual session level, and cancellation is absent.

## Reproducibility

Run `python analysis.py` from this project directory. The script cleans the CSVs, loads them into an in-memory SQLite database, executes `analysis.sql`, exports the user-level table and summaries, and refreshes this report and figures. Python and SQL user-level primary metrics are compared in `results.json`.
