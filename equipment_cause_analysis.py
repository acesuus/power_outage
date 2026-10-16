import csv, re, json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'equipment_analysis'
OUT.mkdir(exist_ok=True)
ROWS = list(csv.DictReader((ROOT / 'paleco_unscheduled_puerto_princesa.csv').open(encoding='utf-8-sig', newline='')))

def clean(raw):
    first = raw.split('\n')[0]
    if 'ACBL Amendments' in raw:
        raw = first + '\n' + raw.rsplit('ACBL Amendments', 1)[1]
    raw = re.split(r'Ang aming paumanhin|Copyright|Para sa katanungan|For inquiries', raw, flags=re.I)[0]
    return re.sub(r'\s+', ' ', raw).strip()

def excerpt(text, match):
    start=max(0,match.start()-30)
    end=min(len(text),match.end()+85)
    if start: start=text.find(' ',start)+1
    if end<len(text): end=text.rfind(' ',match.end(),end)
    return ('...' if start else '')+text[start:end]+('...' if end<len(text) else '')

FAIL = r'(?:na\s*damaged|damaged?|nasira\w*|sirang|sira|nasunog|nagsunog|nasusunog|burn\w*|broken|basag|nabasag|busted|cut|naputol|pagkaputol|nalagot|nalusaw|nagbabaga\w*|bumabaga\w*|umuusok|bulok|nabali\w*|nabaling|pagkabali|nabangang|natumba\w*|natumabang|tumagilid|matutumba\w*|loose|lumuwag|short[ -]?circuit\w*|hot\s*spots?)'
ASSETS = {
    'jumper': r'(?:primary\s+)?jumper(?:\s+wire)?',
    'transformer': r'(?:common\s+|sole(?:\s+use)?\s+)?transformer',
    'fuse_cutout': r'fuse\s*(?:link|cut\s*[ -]?out(?:\s+(?:holder|assembly))?)?|cut\s*[ -]?out(?:\s+assembly)?',
    'insulator': r'(?:(?:pin[ -]?type|suspension|disconnect\s+switch)\s+)?insulator',
    'crossarm': r'cross[ -]?arm',
    'pole': r'(?:distribution\s+)?pole(?!\s+top)|poste',
    'arrester': r'light(?:n?ing)?\s+arrester',
    'connection_hardware': r'connector|connection|CT/PT|pole\s+top\s+pin|hotline\s+clamp|string\s+clamp',
    'switch': r'disconnecting\s+switch|disconnect\s+switch',
    'recloser_hardware': r'(?:LACSAMANA\s+)?recloser',
    'meter': r'metro|meter',
}
RULES = {}
GLUE = r'\s+(?:(?:na|ang|ng|isang|mga|is|was|at|and|the|of|ring|open)\s+){0,3}'
for category, asset in ASSETS.items():
    fail = FAIL
    if category in {'transformer', 'arrester', 'fuse_cutout', 'meter'}:
        fail = '(?:' + FAIL + '|pumutok)'
    RULES[category] = re.compile(r'(?:' + fail + r')\b' + GLUE + r'(?:' + asset + r')\b|\b(?:' + asset + r')\b' + GLUE + r'(?:' + fail + r')\b', re.I)
RULES['line_conductor_cable'] = re.compile(
    r'\b(?:naputol|pagkaputol|nalagot|cut|nasirang|nasira|nasunog|nasusunog|burn\w*|bumagsak|nahulog|nakalaylay|damaged?|masirang|pumutok|umaapoy)\b' + GLUE +
    r'(?:primary\s+(?:line|wire)(?!\s+jumper)|primary(?!\s+(?:jumper|line|wire))|secondary\s+(?:line|wire)(?!\s+jumper)|(?:69\s*kv\s+)?transmission\s+line|tie\s+line|XLPE\s+cable|power\s+cable|neutral\s+line|linya(?:\s+ng\s+kuryente)?|kable|cable|wire(?!\s+jump))\b', re.I)
RULES['breaker_relay_hardware'] = re.compile(
    r'(?:problema|problem|mechanical|mekanikal|damaged?|nasira\w*)\b.{0,60}?\b(?:breaker|relay)\b|\b(?:breaker|relay)\b.{0,40}?\b(?:problema|problem|mechanical|mekanikal|damaged?|nasira\w*)', re.I)
RULES['generation_plant_equipment'] = re.compile(
    r'(?:mechanical\s+problem|mekanikal\s+na\s+problema|problemang\s+mekanikal|technical\s+problem|magka\s*problema|problema|problem)\b.{0,70}?\b(?:unit|planta|plant|generator|makina|PPGI|DMCI|Delta\s*P)\b'
    r'|\b(?:unit|planta|plant|generator|makina|PPGI|DMCI|Delta\s*P)\b.{0,70}?\b(?:mechanical\s+problem|mekanikal\s+na\s+problema|problemang\s+mekanikal|technical\s+problem)\b', re.I)

review = []
# Manually reviewed linguistic exceptions and cause context. These refer to source record numbers.
ADD = {
    'jumper': {381,682,1374},
    'transformer': {561,731},
    'pole': {130,801,971},
    'crossarm': {336},
    'connection_hardware': {205,309},
    'thermal_hotspot': {297,663},
    'line_conductor_cable': {518,955,1094},
    'recloser_protection_settings': {806},
    'breaker_relay_hardware': {875},
    'line_clearance_contact': {18,89,105,115,130,266,270,303,310,350,479,721,727,751,820,940,1123,1179,1374},
}
REMOVE = {'generation_plant_equipment': {785,806,902,952,983,997,1274},
          'line_conductor_cable': {1090,320}, 'fuse_cutout': {1311}}
REPAIR_ONLY = {
    'transformer': {467,611,675,679,696,697,707,878,885,1152},
    'pole': {545,737,911,912,927,1066,1181,1182,1215,1235,1275,1402},
    'jumper': {621,1430,1431},
    'crossarm': {1308},
    'fuse_cutout': {1311},
    'meter': {804},
    'line_conductor_cable': {880,883,1099,1141,1143,1173,1200,1251,927,1181,1182},
    'line_clearance_contact': {800},
}
MANUAL_EVIDENCE = {
    297: 'correct identified hotspots on Structure Nos. 10 and 43',
    663: 'ayusin ang nagbabaga sa linya',
    1094: 'magkumpuni ng nasira/dumikit sa ating linya',
}
for number, row in enumerate(ROWS, 1):
    text = clean(row['raw_text'])
    evidence = {}
    for category, regex in RULES.items():
        matches = list(regex.finditer(text))
        if category == 'line_conductor_cable':
            matches = [m for m in matches if not (re.search(r'nasirang linya',m.group(),re.I)
                       and re.search(r'ibabalik|pagtapos',text[max(0,m.start()-180):m.start()],re.I))]
        if matches:
            evidence[category] = list(dict.fromkeys(excerpt(text,m) for m in matches))
    for category, numbers in REMOVE.items():
        if number in numbers:
            evidence.pop(category, None)
    for category, numbers in ADD.items():
        if number in numbers:
            detail = re.split(r'\b(?:Detalye|Dahilan|Reason)\s*[:;]', text, flags=re.I)[-1]
            evidence[category] = [MANUAL_EVIDENCE.get(number, detail[:1200])]
    repair_categories = [category for category,numbers in REPAIR_ONLY.items() if number in numbers]
    classification = ('equipment_fault_or_damage_stated' if any(c!='line_clearance_contact' for c in evidence)
                      else 'infrastructure_contact_only' if evidence
                      else 'equipment_repair_or_replacement_only' if repair_categories
                      else 'no_specific_equipment_fault_evidence')
    cause_text = re.split(r'\b(?:Detalye|Dahilan|Reason)\s*[:;]', text, flags=re.I)[-1]
    external = []
    if re.search(r'\bpuno\b|\bsanga\b|\bbaging\b|punong|palapa|kawayan|niyog', cause_text, re.I): external.append('vegetation')
    if re.search(r'\bibon\b|\bpusa\b|\btuko\b|\bpaniki\b|kabilaw', cause_text, re.I): external.append('animal')
    if re.search(r'\btruck\b|forklift|tracking|nakabangga|nabangga|nasabitan',cause_text,re.I): external.append('vehicle_or_collision')
    if re.search(r'\bkidlat\b|lightning strike|nakidlatan',cause_text,re.I): external.append('lightning')
    if re.search(r'\btelco\b|Maxtel|street\s*light|drop wire',cause_text,re.I): external.append('third_party_or_streetlight_interface')
    transient = bool(re.search(r'temporary\s+line\s+fault|auto[ -]?reclos',cause_text,re.I))
    trips = bool(re.search(r'(?:nag\s*[ -]?trip|tripping|auto[ -]?tripped|trip ang|cascad)',cause_text,re.I))
    weak = bool(re.search(r'narinig.*pumutok|maaring|posibleng',cause_text,re.I))
    if number == 320: classification = 'possible_line_damage_unconfirmed'
    review.append({'record_number': number, 'event_id': row['event_id'], 'event_date': row['event_date'],
                   'recloser_clean': row['recloser_clean'], 'circuit_clean': row['circuit_clean'],
                   'categories': ';'.join(evidence), 'evidence': json.dumps(evidence, ensure_ascii=False),
                   'repair_only_categories': ';'.join(repair_categories), 'classification':classification,
                   'external_trigger_mentions': ';'.join(external), 'reported_or_probable':int(weak),
                   'temporary_fault_or_auto_reclose':int(transient), 'trip_mentioned':int(trips),
                   'clean_raw_text': text, 'raw_text':row['raw_text'], 'published_at':row['published_at'],
                   'event_date_source':row['event_date_source'], 'post_url':row['post_url'],
                   'is_grid_wide':row['is_grid_wide'], 'is_load_shedding':row['is_load_shedding'],
                   'is_emergency_shutoff':row['is_emergency_shutoff']})
with (OUT/'candidate_review.json').open('w',encoding='utf-8') as f:
    json.dump(review, f, ensure_ascii=False)
print(json.dumps({'candidate_rows':sum(bool(r['categories']) for r in review),
                  'classifications':dict(Counter(r['classification'] for r in review)),
                  'category_counts':dict(Counter(c for r in review for c in r['categories'].split(';') if c))},indent=2))

LABELS = {
    'line_conductor_cable': 'Line, conductor or cable damage', 'jumper':'Jumper failure',
    'pole':'Pole damage or unsafe condition', 'transformer':'Transformer failure',
    'insulator':'Insulator failure', 'generation_plant_equipment':'Generation plant equipment problem',
    'crossarm':'Crossarm failure', 'line_clearance_contact':'Built infrastructure clearance or contact fault',
    'fuse_cutout':'Fuse link or cutout failure', 'connection_hardware':'Connector, clamp, CT/PT or pole pin fault',
    'breaker_relay_hardware':'Breaker or relay hardware problem', 'switch':'Disconnect switch assembly fault',
    'arrester':'Surge/lightning arrester failure', 'thermal_hotspot':'Hotspot or overheating on line/structure',
    'recloser_hardware':'Recloser hardware failure', 'recloser_protection_settings':'Recloser protection settings problem',
    'meter':'Meter failure',
}
fault_rows = [r for r in review if any(c != 'line_clearance_contact' for c in r['categories'].split(';') if c)]
related = [r for r in review if r['categories']]
repair_rows = [r for r in review if r['classification']=='equipment_repair_or_replacement_only']
counts = Counter(c for r in related for c in r['categories'].split(';'))
summary = {
    'source_records':len(review), 'equipment_fault_or_damage_rows':len(fault_rows),
    'equipment_fault_or_damage_percent':round(len(fault_rows)/len(review)*100,2),
    'additional_clearance_contact_only_rows':len(related)-len(fault_rows),
    'broader_infrastructure_related_rows':len(related),
    'broader_infrastructure_related_percent':round(len(related)/len(review)*100,2),
    'repair_replacement_only_rows':len(repair_rows),
    'broader_infrastructure_including_repair_rows':len(related)+len(repair_rows),
    'possible_unconfirmed_line_damage_rows':1,
    'temporary_fault_auto_reclose_rows':sum(r['temporary_fault_or_auto_reclose'] for r in review),
    'trip_without_specific_equipment_fault_rows':sum(r['trip_mentioned'] and not r['categories'] for r in review),
    'reported_or_probable_included_rows':sum(r['reported_or_probable'] for r in related),
    'category_counts':dict(counts),
    'count_unit':'Source rows/notices, not unique physical incidents. Category counts overlap.',
    'classification_method':'English/Filipino fault phrases linked to assets, boilerplate exclusion, and manually reviewed exception cases. Text-supported labels, not field-validated fault diagnoses.',
}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
for r in review:
    categories = set(filter(None,r['categories'].split(';')))
    r['equipment_fault_stated'] = int(any(c != 'line_clearance_contact' for c in categories))
    r['infrastructure_related_stated'] = int(bool(categories))
    r['repair_only'] = int(r['classification']=='equipment_repair_or_replacement_only')
    for category in LABELS:
        r['cause_'+category] = int(category in categories)

features = [
 ['cause_equipment_fault', 'Event annotation', '1 for any named equipment fault/damage, overheating or protection defect; excludes contact-only and repair-only rows.', 'Raw text after outage', 'Use for explanation or as a historical label; same-event use leaks the outcome.'],
 ['cause_<asset_type>', 'Event annotation', 'Separate multi-label flags for line, jumper, transformer, pole, crossarm, insulator, cutout, breaker, recloser and other listed categories.', 'Raw text after outage', 'Category counts overlap. Named affected recloser does not prove the failed asset belongs to that recloser.'],
 ['hist_equipment_fault_days_prev30d/90d/365d', 'Recloser-day', 'Count distinct prior event_date + affected recloser days with stated equipment faults. Exclude grid-wide/upstream plant notices from local counts.', 'Only notices published before prediction-day start in Asia/Manila, with earlier event_date', 'Reporting frequency and repeated updates differ from physical incident frequency; do not call this equipment density.'],
 ['hist_line_damage_days_prev90d/365d', 'Recloser-day', 'Prior distinct days with broken/burned/damaged power conductors or cables.', 'Same publication cutoff; exclude current day', 'A broken line may be caused by a tree or vehicle, rather than asset deterioration.'],
 ['hist_jumper_fault_days_prev90d/365d', 'Recloser-day', 'Prior distinct days with cut/broken/burning jumpers.', 'Same publication cutoff; exclude current day', 'Proxy for observed connection problems; no conductor age/size is available.'],
 ['hist_transformer_fault_days_prev365d', 'Recloser-day', 'Prior distinct days with damaged, burned, shorted or reportedly exploded transformers.', 'Same publication cutoff; exclude current day', 'Keep reported/probable flags. Cannot infer transformer count, loading or nameplate capacity.'],
 ['hist_pole_crossarm_damage_days_prev365d', 'Recloser-day', 'Prior distinct days with broken, rotten, leaning or burned poles/crossarms.', 'Same publication cutoff; exclude current day', 'Preserve collision/tree triggers separately; no pole age/material census exists.'],
 ['hist_insulator_cutout_fault_days_prev365d', 'Recloser-day', 'Prior distinct days with damaged insulators, fuse links or cutouts.', 'Same publication cutoff; exclude current day', 'Asset-condition proxy based on reported faults, not an inventory.'],
 ['hist_days_since_last_equipment_fault', 'Recloser-day', 'Days since the latest previously available fault event; blank plus no-history flag if none.', 'Latest qualifying event must have been published before prediction', 'Do not fill unavailable history with zero.'],
 ['hist_equipment_repair_days_prev90d', 'Recloser-day', 'Prior distinct days with equipment repair/replacement notices, recorded separately from explicit faults.', 'Same publication cutoff; exclude current day', 'A repair notice does not establish an actual component failure.'],
 ['hist_clearance_contact_days_prev365d', 'Recloser-day', 'Prior contacts between power lines and other lines, poles, streetlights, roofs or equipment.', 'Same publication cutoff; exclude current day', 'Interface/clearance proxy; keep vegetation and animal contacts distinct.'],
 ['hist_upstream_equipment_fault_days_prev30d', 'Grid/circuit-day', 'Prior days with plant mechanical faults or grid/substation hardware defects.', 'Only notices available before prediction; use a separate upstream history', 'Avoid attributing a generator failure to local recloser hardware.'],
 ['hist_fault_type_diversity_prev365d', 'Recloser-day', 'Number of different equipment fault categories observed in the preceding 365 days.', 'Same publication cutoff; exclude current day', 'May reflect reporting coverage; small-history counts are unstable.'],
 ['hist_external_trigger_equipment_damage_days_prev365d', 'Recloser-day', 'Prior equipment damage days with manually validated vegetation, animal, vehicle or third-party triggers.', 'Same publication cutoff; exclude current day', 'The exported trigger-mention flags are screening aids; validate causal linkage before modeling.'],
 ['hist_auto_reclose_days_prev30d/90d', 'Recloser-day', 'Prior distinct days mentioning auto-reclose or temporary line fault.', 'Same publication cutoff; exclude current day', 'Electrical disturbance history, not proof that a recloser malfunctioned.'],
 ['hist_known_voltage_class_prior', 'Asset/location history', 'Extract voltage only when explicitly tied to the failed asset/location; retain evidence and unknown values.', 'Use only evidence published before prediction', 'Reported 69 kV / 13.8 kV values are partial observations, not a complete network voltage map.'],
 ['asset_age/count/material/capacity/line_length', 'Needs external inventory', 'Join asset registry, GIS, commissioning, maintenance and loading records.', 'Use measurements valid and available at prediction time', 'Cannot derive these reliably from outage notices alone.'],
]
feature_headers=['feature','grain','definition','availability_rule','limitation']
category_table = [['Equipment/cause','Rows','Typical raw-text evidence']]
for category,count in counts.most_common():
    samples=[json.loads(r['evidence'])[category][0] for r in related if category in r['categories'].split(';')]
    category_table.append([LABELS[category],count,samples[0]])
overview=[['Equipment-related outage text analysis','',''],
          ['Metric','Rows','Interpretation'],
          ['Source records',len(review),'Each CSV row is one notice/update.'],
          ['Stated equipment fault or damage',len(fault_rows),'Includes named component defects, overheating, plant technical faults and protection settings.'],
          ['Additional clearance/contact-only records',len(related)-len(fault_rows),'Power lines touching built infrastructure, without stated component damage.'],
          ['Broader infrastructure-related records',len(related),'Unique rows across fault/damage and built contact categories.'],
          ['Repair/replacement-only records',len(repair_rows),'Excluded from the equipment-fault total.'],
          ['Broad total including repair-only',len(related)+len(repair_rows),'Useful for an intervention history, not a failure count.'],
          ['Possible damage, not established',1,'One possible broken-line report excluded from the stated-fault total.'],
          ['Temporary fault / auto-reclose mentions',summary['temporary_fault_auto_reclose_rows'],'All-source count; overlaps faults and other causes. Reclosing alone is not hardware failure.'],
          ['Trip mentions without specific equipment fault',summary['trip_without_specific_equipment_fault_rows'],'Operational symptom; may be caused by upstream faults, tests, vegetation or unknown causes.'],
          ['Reported/probable included records',summary['reported_or_probable_included_rows'],'Text-supported statements with uncertainty; not field-confirmed diagnoses.'],
          ['Counting rule','','Category counts overlap; source-row totals are not deduplicated physical incidents.'],
          ['Prediction rule','','Use prior published notices only. Same-event causes are outcome information.'],
          ['Location rule','','Clean recloser identifies an affected area; asset ownership/location needs explicit evidence.'],
          ['Inventory limits','','Age, asset density, rated capacity and line length require an asset registry/GIS.'],
          ['Source','','paleco_unscheduled_puerto_princesa.csv; corrected event dates; event_id and record number retained for tracing.']]
record_headers=['Record','Event ID','Event date','Classification','Equipment causes','Repair categories','Trigger mentions','Reported/probable','Recloser affected','Circuit affected','Evidence']
record_rows=[]
for r in review:
    if r['categories'] or r['repair_only_categories']:
        ev=json.loads(r['evidence'])
        snippets='; '.join(category+': '+snips[0] for category,snips in ev.items()) if ev else r['clean_raw_text']
        record_rows.append([r['record_number'],r['event_id'],r['event_date'],r['classification'],r['categories'],r['repair_only_categories'],r['external_trigger_mentions'],r['reported_or_probable'],r['recloser_clean'],r['circuit_clean'],snippets])
csv_headers=['record_number','event_id','event_date','equipment_fault_stated','infrastructure_related_stated','repair_only','classification','categories','repair_only_categories','external_trigger_mentions','reported_or_probable','recloser_clean','circuit_clean','evidence','raw_text','post_url']+[('cause_'+c) for c in LABELS]
csv_matrix=[csv_headers]+[[r[h] for h in csv_headers] for r in related]
all_matrix=[csv_headers]+[[r[h] for h in csv_headers] for r in review]
bundle={'summary':summary,'overview':overview,'categories':category_table,'records':[record_headers]+record_rows,'features':[feature_headers]+features,'related_csv':csv_matrix,'all_flags_csv':all_matrix}
(OUT/'analysis_bundle.json').write_text(json.dumps(bundle,ensure_ascii=False),encoding='utf-8')
print(json.dumps(summary,indent=2))
