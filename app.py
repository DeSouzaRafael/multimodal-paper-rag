import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.ui.app import *  # noqa: F401, F403 — re-export Streamlit app for HF Spaces
