"""Run the recorded-settings analysis and supplemental checks."""
from pathlib import Path
import runpy
import os
ROOT=Path(__file__).resolve().parent
os.chdir(ROOT)
runpy.run_path(str(ROOT/'consistency_analysis_07_10.py'),run_name='__main__')
runpy.run_path(str(ROOT/'submission_checks_09_10.py'),run_name='__main__')
print('Primary and supplemental outputs regenerated. Compare against an untouched archive copy.')
