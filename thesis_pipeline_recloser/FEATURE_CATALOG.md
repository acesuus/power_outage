# Recloser model features

Full family: **88 conceptual inputs**, before categorical encoding/missingness columns. The generated `02_features/feature_dictionary.csv` lists every exact variable and family. Internal `psgc` is a compatibility identity containing the recloser name; it is not a geographic code. The separately saved `recloser` field is the human-readable identifier.

| Group | Inputs | Definition and timing |
|---|---:|---|
| Recloser identity | 1 | Named area; native CatBoost category or fitting-only one-hot encoding |
| Calendar | 6 | Weekday/weekend and cyclical day-of-year/month for D |
| Target-day weather | 14 | Equal-barangay precipitation, temperature max/min/mean, apparent max temperature, humidity mean/max/min, wind speed max/mean, gust max, mean wind sine/cosine and modal weather code |
| Prior weather | 12 | D−1 rainfall/gust/temperature/humidity and 3/7-day precipitation sums, gust maxima, temperature/humidity means; shift before rolling and require complete windows |
| Monthly vegetation means | 6 | `veg_buffer_5m`, `veg_buffer_15m`, `veg_buffer_30m`, `ndvi_p90`, `ndvi_mean_brgy`, `valid_frac_b30`, averaged over available mapped barangays in D's month |
| Outage history | 4 | Prior outage-day counts 3/7 days; recency capped 7; no-outage-in-prior-7 flag |
| Required equipment history | 2 | Prior reported equipment-fault days 7/14; affected-area proxy, not recloser hardware failures |
| Repair history | 2 | Prior repair-notice days 7/14, separate from explicit faults |
| Equipment categories | 8 | Line/cable, jumper, transformer and pole/crossarm fault days, each 7/14 |
| Wet weather history | 3 | Prior wet-day counts 3/7 and wet streak capped 7; threshold 0.1 mm |
| Weather stress | 2 | D mean rainfall/gust divided by per-recloser fitting-only P90, denominator floor 1 |
| Weather × vegetation | 2 | D mean rain/gust × same-month mean 5 m vegetation proxy |
| Rainfall exposure | 2 | Equal-barangay mean precipitation hours and mean daily maximum hourly rainfall, reduced daily source |
| Rain pattern | 4 | D rain presence >0.1 mm; preceding dry streak capped 7; rain after three dry days; D precipitation minus D−1 |
| Rain × history | 5 | Rain flag × outage counts 3/7 and equipment counts 7/14; D mean rain × equipment count 14 |
| Insulator/fuse history | 4 | Prior reported insulator and fuse/cutout fault days 7/14 |
| Other-recloser history | 2 | Distinct prior other-selected-recloser/date pairs 3/7, focal unit excluded; not confirmed connectivity |
| Spatial weather exposure | 6 | Across mapped barangays: rainfall maximum/range, gust maximum/range, precipitation-hours maximum and maximum daily peak-hourly precipitation |
| Spatial vegetation exposure | 3 | Across available mapped barangays: maximum 5/15/30 m buffer vegetation values in D's month |

Weather means are calculated across geographic measurements, never summed across barangays. Temporal rainfall sums then accumulate the mean exposure over earlier calendar days. Spatial peaks describe the most exposed sampled barangay, not a measured feeder corridor. Modal weather codes are categorical; wind degrees are not arithmetically averaged.

History/calendar has **29** inputs. History/weather has **77** inputs. Full has **88** inputs, adding six vegetation means, three vegetation maxima and two weather/vegetation interactions. Outage/equipment histories are reconstructed per recloser; barangay outage counts are not averaged into recloser histories.

All histories exclude D and require publication by D−1 20:00 PHT. Equipment histories use fault-bearing publication, not earlier generic updates. Equipment maximum window is 14 days; prior-weather maximum is 7. Current incident causes/details and affected-area lists do not enter model predictors. Supporting posts, label uncertainty, dates used to derive labels, official URLs, raw text and observed vegetation-barangay fraction are audit metadata.

Missing vegetation remains missing; models use fitting-only medians and missingness indicators. Entirely missing fitting columns are dropped consistently. Same-month vegetation and full-day D/D−1 historical weather are allowed for this retrospective experiment and do not demonstrate advance availability.

Not included: duplicated rain/NDVI fields, unverified vegetation trend/anomaly, physical pole/transformer/recloser ages or inventories, network distances/customer counts/load, circuit polygons or invented service-area coordinates. The frozen proxy mapping and its source evidence are saved separately for review.
