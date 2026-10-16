# Recloser-day suite protocol

## Scope and fixed label evidence

Reuse recloser9_validated_first: 1,369 source posts;430 supporting posts;467 accepted recloser assignments;
418 positive recloser-days on 325 dates;13,473 full-panel rows across nine names. Keep corrected incident
dates/publication fallbacks and the equipment-only Lacsamana assignment exclusion unchanged. The target
is a recorded interruption affecting any part of a named service area, not confirmed failure of that asset.

The existing primary mask excludes otherwise-zero rows on dates with source posts lacking recloser names,
preserving positives. This uses target-day reporting and is conditional/retrospective, not available at warning
time. Preserve the old primary test population: 2,098 rows,57 positives on41 dates; sensitivity includes all
2,376 rows and the same57 positives. Neither mask verifies that no-record rows had uninterrupted power.

## Feature design and spatial proxies

Reuse the frozen mapping learned from incidents through June16,2024 published by June30:33 links across
24 barangays, equal weights within recloser. Do not update mappings with future notices. Weather can be
identical across several units; preserve that fact. No official feeder topology, geographic polygons, line
lengths, customer exposure, coordinates or asset ages are invented.

Compact families adapt the city inputs per recloser and add identity: history9,full37,rainfall22,wind14,
temperature/humidity11,vegetation13,all-weather30,full-local39,full-dry39,full-local-dry41. The legacy-full
family retains the original88 fields as a separate feature benchmark. Every family keeps outage history
3/7 days, recency<=7 and equipment-fault history7/14. Histories use only earlier incidents supported by
publication before D-1 at20:00 PHT; fault categories use their fault-bearing notice availability.

Daily mean/max rain, rainy-site fraction, rain duration/hourly peak, gust mean/max, temperature/humidity
and categorical WMO code summarize mapped barangays. Weather histories span D-1 and preceding3/7days.
Vegetation means and spatial P90 use available mapped barangays in D's month, with missing values retained.
Rain/gust x vegetation local products are formed at the same barangay before taking mean/max, replacing
two cross-area products in local variants. Dry-to-wet fractions are mapped sites rainy on D (>=.1mm) after
each preceding3/7days at that same site was dry. Original recloser wet-day counts retain mapped-mean rain
>=.1mm, not the city's any-site wetness rule. No weather history exceeds7days. Full-D weather and own-month
vegetation remain retrospective and may be unavailable in an operational day-ahead application.

Legacy88 includes repair/fault categories, other-recloser history and fitting-only P90 weather stress.
Its original vegetation valid-fraction predictor remains only for legacy comparability, not as physical
asset condition. Current labels, causes, text, post counts, date/source uncertainty and URLs are excluded
from all model allowlists. Identity fields store names, not official geographic codes.

## Algorithms and transforms

Frozen broad comparison: five algorithms x history/all-weather/full/legacy-full, plus seven remaining
feature families with XGBoost =27base procedures. Monthly broad comparison: logistic/XGBoost x history/full
x12m/24m/expanding; eight remaining compact feature families for12mXGBoost; legacy88XGBoost across all
three windows =23base procedures. RF/CatBoost/spline are frozen baselines, as in the corresponding city
experiments; they are not exhaustively crossed with monthly windows/all feature groups.

Four candidates each: logistic/spline L2C=.01,.1,1,10; XGBoost/CatBoost depth2/3 xL2=10/30; random forest
depth6/10 xminimum leaf10/20 with200trees. Natural class prevalence, no artificial positives/class weighting.
XGBoost child5,learning rate.05,row/column sampling.8; CatBoost learning rate.05,ordered-time settings,
Bernoulli sampling.8; both maximum300trees and early-stopping patience25 on tuning log loss. Seed42,
four threads. Fit-only medians/missing indicators, categorical identity/weekday/WMO. Standardize only
logistic/spline; small spline uses3uniform knots,degree2,constant extrapolation. Legacy P90 ratios are fitted
per unit with global fit-only fallback and denominator floor1. Refit uses selected boosting round count exactly.

Frozen inner tuning minimizes loss, AP/trial ties. Monthly inner tuning maximizes AP, loss/trial ties,
matching their city procedures. These settings differ from some original recloser class-weighted searches;
original models/scores are retained as separate references, not silently replaced by new fits.

## Frozen chronology and selection

Copy the exact original five-role cohort archives for January/April/July/October2025 and January2026.
Expanding history begins September30,2022. Preserve original publication-availability exclusions and
14-day role gaps. Final refit ends September16,2025; calibration ends December17. No2026 outcome enters
frozen tuning/refitting/calibration/selection; learned components and thresholds stay fixed during evaluation.

Compare raw and original nonnegative-slope sigmoid probabilities. Calibration can fit slope0 (constant),
but cannot reverse ranking like four old unconstrained legacy calibrators. Select minimum mean 2025
outer-quarter probability loss, higher AP then ID ties. Save overall/full/raw/AP alternatives before2026.
A formal probability-loss winner is not automatically a useful warning model.

## Monthly chronology, calibration and selection

Origins October2024–September2026, with October–December2024 providing prediction-pool warmup.
At M use records available by M-1 at20:00PHT and mature incident records through M-15. Windows start
M-12/24months (clipped at September30,2022) or expand from that warmup date. Tuning uses M-3months
through M-15; initial fit ends14days before tuning starts. Refit uses the full mature window including tuning.
Exclude unknown late-only positives, never relabel them zero. Apply the existing uncertain-negative mask
to training/calibration, and disclose its conditional nature.

Store predictions for every unit in each target month, including all no-record rows, before knowing its
outcomes. Calibration joins genuinely earlier monthly predictions to known mature calibration row IDs;
require complete joins,>=45rows and both classes. The source has >=7positive calibration rows in all such
windows. Fit intercept-only sigmoid(logit(p)+b), slope1, and compare raw. This preserves monthly ordering
and, with correspondingly retuned cutoffs, learned warning decisions. It does not add predictive information.

Score January1–December17,2025 using truth known by December31 at20:00PHT. Keep December18–31raw
predictions for later calibration, exclude their outcomes from selection. Choose highest pooled development
recall among procedures with pooled FPR<=20%; mean monthlyAP,lower loss,IDties. If none feasible choose
lowestFPR then AP/loss/ID and flag unmet allowance. Also save probability/AP alternatives. For12mXGBoost
feature replacement, additionally require AP improvement over full in>=7/12months and no worse mean loss;
otherwise retain the full reference. This consistency screen is experimental, not an industry standard.
Later2026models may learn completed earlier2026 outcomes after availability/maturity checks. Never fit a
current/future target-month outcome. Frozen/monthly earlier selection criteria differ; report that distinction.

## Warning rules, sensitivity and reporting

For every procedure report five rules learned on separate earlier calibration scores: FPR20/FPR10 maximize
recall within the respective earlier no-record false-positive allowance; balanced accuracy maximizes
(recall+specificity)/2; F1 maximizes precision/recall F1; fixed score>=.50is a separate reference. Highest
threshold wins ties, and no-warning thresholds are allowed. Primary comparison uses FPR20. This does
not guarantee20%futureFPR and is not a utility-approved tolerance. Original legacy warnings used tuning
F1and are explicitly labelled; do not claim threshold changes improve underlying ranking.

Evaluate all procedures on the same2098eligible rows/57positives, not just the best-looking2026candidate.
Apply the same fitted models and cutoffs to2376-row sensitivity. Report per-recloser and monthly counts,
all confusion entries, precision/recall/FPR,AP,AUC,logloss,Brier and accuracy. A zero-warning model's
precision is undefined empirically; the inherited numeric convention stores0. Majority no-outage accuracy
is97.3%with zero recall. All2026results are exploratory because they were previously examined.

Seven-calendar-day moving-block bootstrap with300replicates keeps all unit rows on sampled dates
together. Intervals are conditional on these procedures, not all feature/selection uncertainty. Do not
compare raw accuracy/AP with city/barangay experiments as if targets, prevalence and eligibility matched.
Later unused dates and issue-time weather/vegetation remain necessary for operational confirmation.

## Verification and separate stages

Eight independently executable stages document audit, feature/cohort preparation, frozen development,
frozen selection/final fitting, monthly development, monthly selection/evaluation, frozen/sensitivity/paired
evaluation, and verification/reporting. Nine safeguards cover per-unit publication timing, genuinely earlier
calibration, fitting-only transforms/stress, mandatory features, monotonic calibration, unchanged evaluation
membership and all algorithm input paths. Save687model bundles and2748inner trials with resumable
hash-verified checkpoints. Freeze code/config/source snapshots. Reload every model and check predictions,
fit-only medians/P90, selected rounds and nonnegative calibration. Independently recompute final metrics,
verify original artifact hashes, save completion/manifest and visually inspect exported figures.

References: [thresholds](https://scikit-learn.org/stable/modules/classification_threshold.html),
[calibration](https://scikit-learn.org/stable/modules/calibration.html),
[XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html).
