"""
tests/run_pipeline_10.py

Quick wrapper to call backend.main_service.run_end_to_end_tests()
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.main_service import run_end_to_end_tests

if __name__ == "__main__":
    run_end_to_end_tests()
