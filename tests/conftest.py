"""Test configuration.

The model is the installed `pcrf` package (`pip install -e .`); tests import it
as `from pcrf import data as d`. The only path addition is the apps/training
directory, so the CLI scripts there (e.g. export_onnx.py) can be imported by the
export tests without packaging them.
"""
import sys
from pathlib import Path

_APPS_TRAINING = Path(__file__).resolve().parents[1] / "apps" / "training"
if str(_APPS_TRAINING) not in sys.path:
    sys.path.insert(0, str(_APPS_TRAINING))
