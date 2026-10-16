import csv
import re
import json
from collections import Counter
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'paleco_unscheduled_puerto_princesa.csv'
OUT = ROOT / 'date_verification'
OUT.mkdir(exist_ok=True)
months = {name.lower(): i for i, name in enumerate(
    ['January', 'February', 'March', 'April', 'May', 'June', 'July',
     'August', 'September', 'October', 'November', 'December'], 1)}
months.update({name[:3]: value for name, value in list(months.items())})
months['sept'] = 9
months['devember'] = 12  # Source misspelling, retained verbatim in evidence.
month = '|'.join(sorted(months, key=len, reverse=True))
patterns = [
    (re.compile(r'\b(' + month + r')\.?\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{4})\b', re.I), 'month'),
    (re.compile(r'(?<![\d/])(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})(?!\d)', re.I), 'numeric'),
    (re.compile(r'\b(\d{4})-(\d{2})-(\d{2})\b'), 'iso'),
    (re.compile(r'\b(\d{1,2})\s+(' + month + r')\.?\s*,?\s*(\d{4})\b', re.I), 'day'),
]

def extract(text):
    found = []
    for regex, kind in patterns:
        for match in regex.finditer(text):
            a, b, c = match.groups()
            try:
                if kind == 'month':
                    value = date(int(c), months[a.lower()], int(b))
                elif kind == 'day':
                    value = date(int(c), months[b.lower()], int(a))
                elif kind == 'iso':
                    value = date(int(a), int(b), int(c))
                else:
                    year = int(c) + (2000 if len(c) == 2 else 0)
                    value = date(year, int(a), int(b))
            except ValueError:
                continue
            start = text.rfind('\n', 0, match.start()) + 1
            end = text.find('\n', match.end())
            line = text[start:end if end >= 0 else len(text)].strip()
            prefix = text[start:match.start()].lower()
            if re.search(r'(?:date|time)\s+of\s+incident|oras\s+na\s+nawalan', prefix):
                rank, basis = 0, 'incident_date'
            elif re.search(r'\bpetsa\b|\bdate\s*(?::|;|and duration)', prefix):
                rank, basis = 1, 'explicit_event_date'
            else:
                rank, basis = 2, 'other_raw_text_date'
            found.append({'date': value.isoformat(), 'text': match.group(),
                          'line': line, 'rank': rank, 'basis': basis,
                          'position': match.start()})
    return sorted(found, key=lambda x: x['position'])

with SOURCE.open(encoding='utf-8-sig', newline='') as stream:
    rows = list(csv.DictReader(stream))
audit = []
for number, row in enumerate(rows, 1):
    hits = extract(row['raw_text'])
    dates = sorted({h['date'] for h in hits})
    preferred = min(hits, key=lambda h: (h['rank'], h['position'])) if hits else None
    stored = row['event_date']
    if not hits:
        status = 'unverifiable_no_date_in_raw_text'
        reason = 'No full calendar date found in raw_text; publication date is not independent event-date evidence.'
    elif preferred['date'] != stored:
        if row['date_status'] == 'corrected_source_typo':
            status = 'documented_source_year_correction'
            reason = 'Stored date differs from literal raw_text year; existing note documents correction using publication year.'
        elif preferred['rank'] == 0:
            status = 'mismatch_incident_date'
            reason = 'Stored date uses update date instead of the explicitly stated incident date.'
        else:
            status = 'mismatch_raw_text_date'
            reason = 'Stored event_date differs from the preferred date in raw_text.'
    elif len(dates) > 1:
        if preferred['rank'] == 0:
            status = 'matches_incident_date_separate_update_date'
            reason = 'Stored event_date matches explicit incident/outage start date; later date belongs to update or restoration.'
        elif row['event_id'] in {'4abb3fd72b1580bd', '9266b071b716db8d'}:
            status = 'matches_outage_date_separate_restoration_date'
            reason = 'Outage begins on stored event_date; other date belongs to restoration/update on the next day. Context manually reviewed.'
        else:
            status = 'matches_event_date_conflicting_title'
            reason = 'Petsa and event_date agree, but title contains a different date; likely stale title. Context manually reviewed.'
    elif any('devember' in h['text'].lower() for h in hits):
        status = 'matches_after_month_spelling_normalization'
        reason = 'Raw text spells December as Devember; day and year agree. Month typo manually reviewed.'
    else:
        status = 'matches_raw_text'
        reason = 'Stored event_date matches the date in raw_text.'
    audit.append({
        'record_number': number, 'event_id': row['event_id'], 'event_date': stored,
        'preferred_raw_date': preferred['date'] if preferred else '',
        'date_basis': preferred['basis'] if preferred else '',
        'recommended_event_date': preferred['date'] if status.startswith('mismatch_') else '',
        'audit_status': status, 'reason': reason,
        'raw_dates': ';'.join(dates),
        'date_evidence': json.dumps(hits, ensure_ascii=False),
        'event_date_source': row['event_date_source'], 'existing_date_status': row['date_status'],
        'existing_date_note': row['date_note'], 'published_at': row['published_at'],
        'post_url': row['post_url'], 'title': row['title'], 'raw_text': row['raw_text'],
    })
for filename, data in [('event_date_audit.csv', audit),
                       ('event_date_exceptions.csv', [r for r in audit if r['audit_status'] != 'matches_raw_text'])]:
    with (OUT / filename).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(data)
summary = {'total_records': len(audit), 'status_counts': dict(Counter(r['audit_status'] for r in audit))}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
for record in audit:
    if record['audit_status'] not in ['matches_raw_text', 'unverifiable_no_date_in_raw_text']:
        print(json.dumps({k: record[k] for k in ['record_number', 'event_id', 'event_date', 'preferred_raw_date', 'audit_status', 'date_evidence']}, ensure_ascii=True))
print('Previously inferred dates now found:', json.dumps([
    {k: r[k] for k in ['record_number', 'event_id', 'event_date', 'preferred_raw_date', 'audit_status']}
    for r in audit if r['event_date_source'] == 'publication_datetime' and r['raw_dates']], indent=2))
