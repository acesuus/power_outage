"""Download daily Open-Meteo weather for barangays appearing in the outage logs.

Python 3.10+, standard library only. Run from any directory. Source CSVs are read
only. Defaults follow the existing daily weather dates, extended if needed to
cover earlier outage dates. Coordinates are published boundary-derived points,
not surveyed barangay halls or weather stations.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import math
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVISION = "e360c9da26ae9455f357ab7d048389a1a8f05c2a"
COORDINATE_URL = (
    "https://raw.githubusercontent.com/fish-and-bear/psgc/"
    f"{SOURCE_REVISION}/psgc/data/core/barangays.json"
)
COORDINATE_DOCUMENTATION = (
    f"https://github.com/fish-and-bear/psgc/blob/{SOURCE_REVISION}/README.md"
)
BOUNDARY_SOURCE = "https://data.humdata.org/dataset/cod-ab-phl"
CITY_CODE = "1731500000"
API_URL = "https://archive-api.open-meteo.com/v1/archive"
API_DOCUMENTATION = "https://open-meteo.com/en/docs/historical-weather-api"
SAN_CARLOS_SOURCE = (
    "https://www.deped.gov.ph/wp-content/uploads/2021/06/"
    "SY-2020-2021-Masterlist-of-Schools-Address-only-1-1.pdf"
)
VALID_COORDINATE_SOURCES = {
    "hdx_exact_2023", "hdx_correspondence_2023", "hdx_name_parent_2023"
}
EXPECTED_UNITS = {
    "weather_code": "wmo code",
    "temperature_2m_max": "°C",
    "temperature_2m_min": "°C",
    "temperature_2m_mean": "°C",
    "apparent_temperature_max": "°C",
    "relative_humidity_2m_mean": "%",
    "relative_humidity_2m_max": "%",
    "relative_humidity_2m_min": "%",
    "wind_speed_10m_max": "km/h",
    "wind_speed_10m_mean": "km/h",
    "wind_gusts_10m_max": "km/h",
    "wind_direction_10m_dominant": "°",
    "precipitation_sum": "mm",
    "rain_sum": "mm",
}
META_COLUMNS = [
    "psgc", "psgc_current", "barangay", "date", "year_month",
    "requested_latitude", "requested_longitude", "api_latitude",
    "api_longitude", "elevation_m", "timezone", "weather_model_selection",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, columns: list[str], rows) -> None:
    # Replace the final file only after the complete output has been written.
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def normalize_name(name: str) -> str:
    name = name.casefold().strip()
    name = re.sub(r"\s*\(pob(?:lacion)?\.?\)", "", name)
    name = re.sub(r"\bsta\.?\s+", "santa ", name)
    name = re.sub(r"[^a-z0-9]+", " ", name).strip()
    if name == "barangay ng mga mangingisda":
        return "mangingisda"
    return name


def reference_weather(path: Path):
    # Open-Meteo's downloaded CSV has a metadata block before the daily table.
    with path.open(encoding="utf-8-sig", newline="") as handle:
        lines = list(csv.reader(handle))
    start = next((i for i, row in enumerate(lines) if row and row[0] == "time"), None)
    if start is None:
        raise ValueError("Reference weather CSV has no daily 'time' header.")
    header = lines[start]
    variables = []
    for label in header[1:]:
        variable = label.split(" (", 1)[0]
        if variable == "precipitation_hours":
            continue
        if variable not in EXPECTED_UNITS:
            raise ValueError(f"Unrecognized daily reference column: {label}")
        expected_label = f"{variable} ({EXPECTED_UNITS[variable]})"
        if label != expected_label:
            raise ValueError(f"Unexpected reference units: {label}")
        variables.append((variable, label))
    if len({key for key, label in variables}) != len(variables):
        raise ValueError("Duplicate reference weather variables.")
    dates = [date.fromisoformat(row[0]) for row in lines[start + 1:] if row]
    if not dates or not variables:
        raise ValueError("Reference weather CSV has no daily records or variables.")
    return variables, min(dates), max(dates)


def request_json(url: str, attempts: int = 4):
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "PALECO-thesis-daily-weather/1.0"})
            with urllib.request.urlopen(request, timeout=45) as response:
                raw = response.read()
            result = json.loads(raw)
            if isinstance(result, dict) and result.get("error"):
                raise ValueError(f"API error: {result.get('reason', result.get('error'))}")
            return result, hashlib.sha256(raw).hexdigest()
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")[:500]
            if exc.code == 429 and any(word in message.casefold() for word in ["hourly", "daily"]):
                raise RuntimeError(f"Open-Meteo quota: {message}. Completed requests are cached; rerun after the quota resets.") from exc
            if exc.code not in {429, 500, 502, 503, 504} or attempt == attempts - 1:
                raise RuntimeError(f"HTTP {exc.code}: {message}") from exc
            delay = min(60, max(2 ** (attempt + 1), int(exc.headers.get("Retry-After", "0")) if exc.headers.get("Retry-After", "0").isdigit() else 0))
            if exc.code == 429:
                delay = 60
            print(f"Transient HTTP {exc.code}: {message}", flush=True)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == attempts - 1:
                raise RuntimeError(f"Download failed: {exc}") from exc
            delay = min(30, 2 ** (attempt + 1))
        print(f"Request retry {attempt + 1}/{attempts - 1} after {delay}s", flush=True)
        time.sleep(delay)
    raise AssertionError("Unreachable retry state")


def coordinate_reference(cache: Path):
    path = cache / "puerto_princesa_coordinate_reference.json"
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("source_revision") != SOURCE_REVISION:
            raise ValueError("Coordinate cache source revision differs from this script.")
        return result
    print("Downloading the version-pinned barangay coordinate reference...", flush=True)
    records, digest = request_json(COORDINATE_URL)
    records = [row for row in records if row.get("city_code") == CITY_CODE]
    if len(records) != 66:
        raise ValueError(f"Expected 66 Puerto Princesa reference barangays, found {len(records)}.")
    result = {
        "source_url": COORDINATE_URL, "source_revision": SOURCE_REVISION,
        "source_file_sha256": digest, "boundary_source": BOUNDARY_SOURCE,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "records": records,
    }
    write_json(path, result)
    return result


def unique_lookup(rows, field: str):
    lookup = {}
    for row in rows:
        key = normalize_name(row[field])
        if key in lookup and lookup[key] != row:
            raise ValueError(f"Ambiguous normalized name: {row[field]}")
        lookup[key] = row
    return lookup


def build_coordinates(log_rows, vegetation_rows, reference, all_barangays=False):
    vegetation = {}
    for row in vegetation_rows:
        key = normalize_name(row["barangay"])
        pair = {"barangay": row["barangay"], "psgc": row["psgc"]}
        if key in vegetation and vegetation[key] != pair:
            raise ValueError(f"Ambiguous vegetation identifiers: {row['barangay']}")
        vegetation[key] = pair
    source = unique_lookup(reference["records"], "name")
    mentions = collections.Counter(
        name.strip() for row in log_rows for name in row["barangay_clean"].split(";") if name.strip()
    )
    mapping = []
    selected = set(vegetation) if all_barangays else set()
    for name, count in sorted(mentions.items()):
        key = normalize_name(name)
        method = "standardized_barangay_name"
        mapping_source = COORDINATE_DOCUMENTATION
        if key == "san carlos":
            key = "bacungan"
            method = "sitio_to_parent_barangay"
            mapping_source = SAN_CARLOS_SOURCE
        if key not in vegetation or key not in source:
            raise ValueError(f"Unmatched outage location: {name}. Resolve its mapping before downloading.")
        selected.add(key)
        mapping.append({
            "outage_location_name": name, "barangay": vegetation[key]["barangay"],
            "psgc": vegetation[key]["psgc"], "psgc_current": source[key]["psgc_code"],
            "rows_mentioning_location": count, "mapping_method": method,
            "mapping_source": mapping_source,
        })
    coordinates = []
    for key in sorted(selected):
        record = source.get(key)
        if record is None:
            raise ValueError(f"No coordinate reference for {vegetation[key]['barangay']}.")
        point = record.get("coordinate") or {}
        origin = record.get("coordinate_source")
        if origin not in VALID_COORDINATE_SOURCES:
            raise ValueError(f"Unverified coordinate source for {record['name']}: {origin}")
        latitude, longitude = point.get("latitude"), point.get("longitude")
        if latitude is None or longitude is None or not (9.3 <= latitude <= 10.6 and 118.1 <= longitude <= 119.3):
            raise ValueError(f"Missing/out-of-area coordinate for {record['name']}: {point}")
        aliases = [row["outage_location_name"] for row in mapping if row["psgc_current"] == record["psgc_code"]]
        coordinates.append({
            **vegetation[key], "psgc_current": record["psgc_code"],
            "official_name": record["name"], "latitude": latitude, "longitude": longitude,
            "outage_location_aliases": ";".join(aliases),
            "coordinate_method": "published_HDX_boundary_derived_representative_point",
            "coordinate_source": origin, "coordinate_source_url": COORDINATE_URL,
            "source_revision": SOURCE_REVISION,
        })
    return coordinates, mapping


def date_range(start: date, end: date):
    return [(start + timedelta(days=i)).isoformat() for i in range((end - start).days + 1)]


def year_chunks(start: date, end: date):
    while start <= end:
        stop = min(end, date(start.year, 12, 31))
        yield start, stop
        start = stop + timedelta(days=1)


def validate_response(response, expected_dates, variables):
    if response.get("utc_offset_seconds") != 28800 or response.get("timezone") != "Asia/Manila":
        raise ValueError("Weather response is not in Asia/Manila (UTC+8).")
    if "hourly" in response:
        raise ValueError("Unexpected hourly data in a daily-only download.")
    daily = response.get("daily", {})
    if daily.get("time") != expected_dates:
        raise ValueError("Weather response dates are missing, duplicated, or out of order.")
    for key, label in variables:
        if response.get("daily_units", {}).get(key) != EXPECTED_UNITS[key]:
            raise ValueError(f"Unexpected API units for {key}.")
        values = daily.get(key)
        if not isinstance(values, list) or len(values) != len(expected_dates):
            raise ValueError(f"Missing or wrong-length daily variable: {key}")
        if any(value is None or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
            raise ValueError(f"Missing/non-numeric weather values in {key}; output will not be declared complete.")
    for key in ["latitude", "longitude", "elevation"]:
        value = response.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Missing API coordinate/elevation: {key}")


def fetch_weather(coordinates, variables, start, end, model, batch_size, cache, pause, workers=3, refresh=False):
    rows = []
    manifest = []
    chunks = list(year_chunks(start, end))
    batches = [coordinates[i:i + batch_size] for i in range(0, len(coordinates), batch_size)]
    jobs = []
    for chunk_start, chunk_end in chunks:
        expected_dates = date_range(chunk_start, chunk_end)
        for batch_number, batch in enumerate(batches, 1):
            params = {
                "latitude": ",".join(str(row["latitude"]) for row in batch),
                "longitude": ",".join(str(row["longitude"]) for row in batch),
                "start_date": chunk_start.isoformat(), "end_date": chunk_end.isoformat(),
                "daily": ",".join(key for key, label in variables), "timezone": "Asia/Manila",
                "temperature_unit": "celsius", "wind_speed_unit": "kmh",
                "precipitation_unit": "mm", "cell_selection": "land",
            }
            if model != "best_match":
                params["models"] = model
            url = API_URL + "?" + urllib.parse.urlencode(params)
            cache_key = hashlib.sha256(url.encode()).hexdigest()
            cached = cache / (cache_key + ".json")
            jobs.append((chunk_start.year, batch_number, batch, expected_dates, url, cached))

    def fetch_batch(job):
        year, batch_number, batch, expected_dates, url, cached = job
        from_cache = cached.exists() and not refresh
        if not from_cache:
            print(f"Fetching {year}, batch {batch_number}/{len(batches)} ({len(batch)} barangays)", flush=True)
        if from_cache:
            envelope = json.loads(cached.read_text(encoding="utf-8"))
            if envelope.get("request_url") != url:
                raise ValueError("Weather cache does not match the request.")
        else:
            response, digest = request_json(url)
            envelope = {
                "request_url": url, "response_sha256": digest,
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "response": response,
            }
        responses = envelope["response"]
        responses = responses if isinstance(responses, list) else [responses]
        if len(responses) != len(batch):
            raise ValueError("Number of API locations differs from the coordinate batch.")
        result_rows = []
        for index, (coordinate, response) in enumerate(zip(batch, responses)):
            if "location_id" in response and response["location_id"] != index:
                raise ValueError("Unexpected location ordering in API response.")
            validate_response(response, expected_dates, variables)
            for i, timestamp in enumerate(expected_dates):
                row = {
                    "psgc": coordinate["psgc"], "psgc_current": coordinate["psgc_current"],
                    "barangay": coordinate["barangay"], "date": timestamp,
                    "year_month": timestamp[:7], "requested_latitude": coordinate["latitude"],
                    "requested_longitude": coordinate["longitude"],
                    "api_latitude": response["latitude"], "api_longitude": response["longitude"],
                    "elevation_m": response["elevation"], "timezone": "Asia/Manila",
                    "weather_model_selection": model,
                }
                row.update({label: response["daily"][key][i] for key, label in variables})
                result_rows.append(row)
        if not from_cache:
            # Cache only after integrity checks pass, so bad responses can be retried.
            write_json(cached, envelope)
            time.sleep(pause)
        result_manifest = {
            "request_url": url, "cache_file": cached.name,
            "psgc_current_in_response_order": [row["psgc_current"] for row in batch],
            "retrieved_at_utc": envelope["retrieved_at_utc"],
            "response_sha256": envelope["response_sha256"],
        }
        return result_rows, result_manifest, from_cache

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_batch, job) for job in jobs]
        try:
            for completed, future in enumerate(as_completed(futures), 1):
                result_rows, result_manifest, from_cache = future.result()
                rows.extend(result_rows)
                manifest.append(result_manifest)
                print(f"Validated {completed}/{len(jobs)} batches" + (" (cached)" if from_cache else ""), flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    manifest.sort(key=lambda record: record["request_url"])
    rows.sort(key=lambda row: (row["psgc_current"], row["date"]))
    pairs = {(row["psgc_current"], row["date"]) for row in rows}
    expected_count = len(coordinates) * len(date_range(start, end))
    if len(rows) != expected_count or len(pairs) != expected_count:
        raise ValueError("Final weather table has missing or duplicate barangay-date rows.")
    return rows, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outages", type=Path, default=ROOT / "paleco_unscheduled_puerto_princesa_city_only.csv")
    parser.add_argument("--vegetation", type=Path, default=ROOT / "vegetation_features_monthly.csv")
    parser.add_argument("--reference-weather", type=Path, default=ROOT / "daily_weather_data.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    parser.add_argument("--model", default="best_match", choices=["best_match", "era5", "era5_seamless", "ecmwf_ifs"])
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--pause-seconds", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=3, help="Maximum concurrent requests (1..3).")
    parser.add_argument("--refresh-weather", action="store_true", help="Re-download weather instead of using cached responses.")
    parser.add_argument("--all-barangays", action="store_true", help="Include all 66 vegetation barangays instead of only those in the outage logs.")
    parser.add_argument("--coordinates-only", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= 10 or args.pause_seconds < 0 or not 1 <= args.workers <= 3:
        parser.error("batch-size must be 1..10, workers 1..3, and pause-seconds nonnegative")
    logs = read_csv(args.outages)
    vegetation = read_csv(args.vegetation)
    variables, reference_start, reference_end = reference_weather(args.reference_weather)
    incident_dates = [date.fromisoformat(row["event_date"]) for row in logs if row["event_date"]]
    start = args.start_date or min(reference_start, min(incident_dates))
    end = args.end_date or max(reference_end, max(incident_dates))
    if start > end or end > datetime.now(timezone(timedelta(hours=8))).date():
        parser.error("Dates must be ordered and cannot include future days.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / "cache"
    cache.mkdir(exist_ok=True)
    reference = coordinate_reference(cache)
    coordinates, mapping = build_coordinates(logs, vegetation, reference, args.all_barangays)
    coordinate_path = args.output_dir / "barangay_coordinates.csv"
    mapping_path = args.output_dir / "barangay_name_mapping.csv"
    write_csv(coordinate_path, list(coordinates[0]), coordinates)
    write_csv(mapping_path, list(mapping[0]), mapping)
    print(f"Matched {len(mapping)} outage location names to {len(coordinates)} barangays. Dates: {start} to {end}.", flush=True)
    if args.coordinates_only:
        return 0
    rows, requests = fetch_weather(coordinates, variables, start, end, args.model, args.batch_size, cache, args.pause_seconds, args.workers, args.refresh_weather)
    weather_path = args.output_dir / "barangay_daily_weather.csv"
    write_csv(weather_path, META_COLUMNS + [label for key, label in variables], rows)
    signatures = collections.defaultdict(list)
    for coordinate in coordinates:
        block = [row for row in rows if row["psgc_current"] == coordinate["psgc_current"]]
        values = [[row[label] for key, label in variables] for row in block]
        digest = hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()
        signatures[digest].append(coordinate["barangay"])
    summary = {
        "complete": True, "start_date": str(start), "end_date": str(end),
        "barangays": len(coordinates), "days_per_barangay": len(date_range(start, end)),
        "rows": len(rows), "daily_weather_variables": [key for key, label in variables],
        "excluded_reference_variable": "precipitation_hours", "hourly_variables_requested": [],
        "timezone": "Asia/Manila", "model_selection": args.model,
        "coordinate_source": COORDINATE_URL, "source_revision": SOURCE_REVISION,
        "coordinate_source_categories": dict(collections.Counter(row["coordinate_source"] for row in coordinates)),
        "input_sha256": {str(path.resolve()): sha256(path) for path in [args.outages, args.vegetation, args.reference_weather]},
        "output_sha256": {path.name: sha256(path) for path in [coordinate_path, mapping_path, weather_path]},
        "missing_weather_values": 0, "duplicate_barangay_date_pairs": 0,
        "distinct_full_weather_series": len(signatures),
        "barangays_with_identical_full_weather_series": [names for names in signatures.values() if len(names) > 1],
        "requests": requests,
    }
    write_json(args.output_dir / "download_summary.json", summary)
    print(json.dumps({key: summary[key] for key in ["complete", "barangays", "days_per_barangay", "rows", "daily_weather_variables", "distinct_full_weather_series"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
