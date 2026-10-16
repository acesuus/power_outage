# Daily weather by barangay

Run with Python 3.10 or newer. No third-party packages or API key are required for
the public non-commercial endpoints used here.

```powershell
python .\weather_data\fetch_barangay_weather.py
```

The script reads the city-only outage logs, identifies the included barangays,
matches them to the monthly vegetation identifiers, and downloads daily weather
for each matched coordinate. It does not modify the original source files or the
previous recloser dataset.

Outputs are saved beside the script:

- `barangay_coordinates.csv`: one published representative weather-request point
  per included barangay, with coordinate provenance and both identifier systems.
- `barangay_name_mapping.csv`: each extracted location name, its standardized
  barangay, and the source for the mapping.
- `barangay_daily_weather.csv`: one row per barangay per local calendar date.
- `download_summary.json`: coverage checks, source/output hashes, request URLs,
  retrieval dates, model selection, and identical weather-series groups.
- `cache/`: the version-pinned coordinate reference and raw API response envelopes
  used to resume downloads without repeating completed requests.

The 14 weather variables retain the names and units from the existing daily
weather CSV. `precipitation_hours (h)` is omitted. Although this variable measures
rainy hours within a day rather than an hourly time series, it was removed at the
requester's direction. No `hourly=` variables are requested.

Defaults use `Asia/Manila`, Celsius, km/h, millimeters, and Open-Meteo's default
`best_match` historical model selection. Requests are split by calendar year and
batched across five locations, using up to three concurrent requests, caching,
and retry handling. Use `--workers 1` for sequential requests or
`--refresh-weather` to replace cached weather responses. Missing or
malformed responses stop the run instead of producing a falsely complete file.
The final weather CSV is replaced only after every requested location/date has
been validated. A failed run may leave coordinate tables and cached responses.

The free API limits are 600 weighted calls/minute, 5,000/hour, and 10,000/day.
Long date ranges and more than ten variables count as multiple calls, so a full
58-barangay download can cross an hourly limit even though the script sends only
60 HTTP requests. On a minute limit it waits 60 seconds before retrying. On an
hourly/daily limit it stops with a clear message. Rerun the same command after
the relevant limit resets; validated cached batches are reused. See
[Open-Meteo call accounting and limits](https://open-meteo.com/en/pricing).

By default, coverage spans the dates in the existing weather file and extends to
any outage dates outside that range. For the present inputs this is August 31,
2022 through October 9, 2026. Change dates explicitly to include history needed
for rolling features, for example:

```powershell
python .\weather_data\fetch_barangay_weather.py --start-date 2022-08-01 --end-date 2026-10-09
```

Other options:

```powershell
# Download for all 66 vegetation barangays, including those absent from the logs.
python .\weather_data\fetch_barangay_weather.py --all-barangays --output-dir .\weather_data\all_66

# Build the mapping and coordinate CSVs without downloading weather.
python .\weather_data\fetch_barangay_weather.py --coordinates-only

# Request one explicitly selected historical model instead of best_match.
python .\weather_data\fetch_barangay_weather.py --model ecmwf_ifs --output-dir .\weather_data\ifs
```

## Coordinates and identifiers

Coordinates come from the community-maintained
[psgc reference](https://github.com/fish-and-bear/psgc), pinned to commit
`e360c9da26ae9455f357ab7d048389a1a8f05c2a`. It matches current PSA PSGC names/codes to
[November 2023 HDX/OCHA administrative boundaries](https://data.humdata.org/dataset/cod-ab-phl)
derived from PSA/NAMRIA. The 66 Puerto Princesa entries have the provenance
category `hdx_correspondence_2023`. The script rejects unverified fallback points.
These are boundary-derived representative coordinates, not surveyed barangay
hall coordinates. Their polygon containment and population representativeness
have not been independently checked against the full boundary shapefiles.

`psgc` preserves the identifier in the vegetation CSV (such as `PH1705316002`) so
that the files can be joined directly. `psgc_current` is the current ten-digit PSGC
identifier from the coordinate reference (such as `1731500002`). Do not assume
the older vegetation identifiers are current ten-digit PSGC codes.

The script standardizes punctuation, `Sta.`/`Santa`, and `(Pob.)` suffixes, and
links the older `Barangay ng mga Mangingisda` name to `Mangingisda`. San Carlos
is mapped to Bacungan because it is a sitio within Bacungan. This is supported
by the [DepEd school address masterlist](https://www.deped.gov.ph/wp-content/uploads/2021/06/SY-2020-2021-Masterlist-of-Schools-Address-only-1-1.pdf),
which lists San Carlos Elementary School at Sitio San Carlos, Bacungan, Puerto
Princesa City. Location mappings do not validate the outage labels themselves.

## Weather interpretation and joins

Open-Meteo historical weather is model/reanalysis data, not measurements from a
separate station in each barangay. Nearby locations may share the same model
grid cell or have identical weather series. `requested_latitude/longitude` retain
the barangay reference point; `api_latitude/longitude` retain the grid coordinates
returned by the API. The default `best_match` can combine models with different
resolutions and does not imply 58 independent weather stations.

- Join outage labels to weather using the standardized barangay identifier and
  local event date.
- Join monthly vegetation using `psgc` and the latest available month. The
  `year_month` weather key assists retrospective joins, but a completed current
  month's satellite composite is not automatically available earlier that month.
- For next-day forecasting, use weather known before the prediction cutoff or
  archived forecasts issued before it. Actual weather observed during the target
  day must not be presented as a forecast input available the day before.

API definitions, variable units, model choices, and attribution:
[Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api).

Offline integrity checks:

```powershell
python -m unittest discover -s .\weather_data -p test_fetch_barangay_weather.py
```
