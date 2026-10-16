"""V2 helpers; reuse immutable V1 metrics/calibration without editing V1 files."""
from pathlib import Path
import sys

V1 = Path(__file__).resolve().parents[2] / 'thesis_pipeline'
if str(V1) not in sys.path:
    sys.path.append(str(V1))
