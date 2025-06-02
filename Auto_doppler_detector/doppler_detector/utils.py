# File: doppler_detector/utils.py
import re
from datetime import datetime


def parse_flac_time_from_filename(filepath: str):
    """
    Given a filename of the form:
       <camera_name> - YYYY-MM-DD HH-MM-SS - ... .flac
    extract the date/time portion and return a datetime object.
    If no date/time is found, raises a ValueError.
    """
    basename = filepath.split("/")[-1]
    pattern = r"([0-9]{4}-[0-9]{2}-[0-9]{2})[ _-]+([0-9]{2}-[0-9]{2}-[0-9]{2})"
    m = re.search(pattern, basename)
    if not m:
        raise ValueError(f"No valid date/time substring found in {basename}")
    date_str = m.group(1)
    time_str = m.group(2)
    dt_str = f"{date_str} {time_str.replace('-', ':')}"
    return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
