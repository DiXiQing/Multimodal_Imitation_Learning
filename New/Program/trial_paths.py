"""Discover timestamp-named grasp trials and legacy numbered trials."""
from pathlib import Path
import re


def grasp_folders(source):
    return sorted(p for p in Path(source).iterdir() if p.is_dir()
                  and (re.fullmatch(r'grasp_\d{3}',p.name) or re.fullmatch(r'\d{8}_\d{6}',p.name)
                       or re.fullmatch(r'(?:Black|Red|Blue)_\d+(?:\.\d+)?cm_\d{8}_\d{6}',p.name)
                       or p.name.startswith('TRIAL_'))
                  and (p/'camera.csv').is_file() and (p/'segment.json').is_file())
