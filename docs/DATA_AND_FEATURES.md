# Data, labels and candidate inputs

## Interruption records and labels

The exact candidate source is `paleco_unscheduled_puerto_princesa_city_only.csv`
(1,369 posts). Upstream and earlier cleaned CSVs are supplied for traceability;
they are not interchangeable with that source. A later audited 1,364-post copy
is outside this snapshot and was not used to train these candidates.

The panel has 13,473 recloser-days and 418 positive recloser-days over its full
history. One post can name several reclosers, and several posts can describe the
same recloser-day. A positive row is a recloser-day, not a unique post or outage
incident. Explicit accepted recloser assignments are recorded in the reference
run's `01_audit/recloser_outage_evidence.csv` and assignment-review files.

Incident dates take precedence when stated; the requested publication-date fallback
is retained where no incident date can be extracted. Date uncertainty and source
URLs remain in source/evidence files. An explicit equipment/fault-location mention
is not unconditionally treated as an affected recloser; consult the accepted
assignment evidence rather than reconstructing labels from name matches alone.

Nine reclosers: BUNCAG, CHARIOT, FACTOR, KALIKASAN, LACSAMANA, MENDOZA, PENRO,
PILTEL and SAN MANUEL. The frozen map has 33 recloser/barangay links over 24
barangays and was constructed from earlier text evidence. It is a geographic
proxy for aggregation, not an official service-area GIS map.

The primary 2026 test contains 2,098 eligible rows: 57 positive and 2,041
no-record rows, across 41 positive calendar dates. The all-no-record sensitivity
population has 2,376 rows and the same 57 positives. Dates with an unlocalized
outage report can leave recloser negatives uncertain; the retrospective primary
mask excludes those uncertain negatives. A model cannot know that target-day
mask prospectively. Reporting is conditional on this evaluation definition.

## Exact history inputs

| Raw input | Meaning / timing |
|---|---|
| `recloser_id` | Fixed recloser identity; one-hot encoded |
| `day_of_week` | Target-day weekday; one-hot encoded |
| `day_of_year_sin` | Annual seasonality, target calendar day |
| `day_of_year_cos` | Annual seasonality, target calendar day |
| `outage_days_prev3d` | Recorded outage days in the previous three days |
| `outage_days_prev7d` | Recorded outage days in the previous seven days |
| `days_since_last_outage_capped7` | Time since last recorded outage, capped at seven days |
| `equipment_fault_days_prev7d` | Prior seven days with equipment-related outage evidence |
| `equipment_fault_days_prev14d` | Prior fourteen days with equipment-related outage evidence |

Prior-day windows exclude the target day's label and use only qualifying posts
published by **20:00 PHT on D-1**. Equipment flags come from the
included raw-text flag dataset and feature dictionary. Counts describe recorded
equipment-related days, not transformer age, pole age or a surveyed asset count.
Missing-value imputation and category encoding are fitted on the training rows.
The nine raw fields expand into more encoded columns.

The candidate uses the archived engineered values. The original
`thesis_pipeline/lib/features.py::history_features` checks both earlier incident
dates and publication availability at each row's cutoff. Equipment history uses
the publication of the fault-bearing post, so an earlier update without fault
details cannot make the fault available early. The package verifier independently
rebuilds these five history fields from the supplied evidence and equipment flags.
Separate publication/maturity filters apply to fitting and calibration cohorts.

## Weather, vegetation and other engineered features

Per-barangay daily weather, original city weather, daily rainfall aggregates,
monthly vegetation, coordinates and name mappings are supplied. Consult their
download scripts/manifests and the feature dictionaries for units, weather
variables, accumulation definitions and spatial aggregation. Rainfall hourly
source measurements were aggregated into daily features; this archive does not
change the task to hourly prediction.

The full suite panel also contains weather/vegetation and interaction columns.
`02_features/families.json` is the authoritative feature-family allowlist.
`02_features/feature_dictionary.csv` plus the reference feature dictionary and
`FEATURE_CATALOG.md` explain them. Their presence in the CSV does not mean that
the history model consumes them. Target-day historical weather and same-month
vegetation in other families are retrospective conditioning variables, not
proof of day-ahead availability. No forecast-weather claim is made here.

The original city weather CSV is an additional reference, not the per-barangay
weather source used to construct the suite panel. Keep its original schema;
use the engineered panel/dictionaries when selecting daily model variables.
