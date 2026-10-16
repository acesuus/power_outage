"""Reuse audited prior utilities without changing their source."""
import sys
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]
for path in [PROJECT / 'thesis_pipeline', PROJECT / 'thesis_pipeline_v2']:
    if str(path) not in sys.path:
        sys.path.append(str(path))
