# app/utils/time_parse.py
import re
from datetime import datetime

_DATE_RE = re.compile(
    r"""^
        \s*pixel\s*-\s*(?P<pixel>\d+)\s*-\s*
        (?P<start>\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}-\d{2}-\d{2})\s*-\s*
        (?P<end>\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}-\d{2}-\d{2})
        (?:\.\w+)?  # Optional file extension
        \s*$
    """, re.X
)

def _smart_parse_date(s: str) -> datetime:
    """
    Accepts:
      - YYYY-DD-MM HH-MM-SS  (e.g., 2025-19-08 10-19-59 where 19 is day, 08 is month)
      - YYYY-MM-DD HH-MM-SS  (e.g., 2025-05-29 10-46-00 where 05 is month, 29 is day)
    Returns naive datetime (local clock, no tz).
    """
    date_part, time_part = s.split()
    year_str, middle, last = date_part.split("-")
    hour_str, min_str, sec_str = time_part.split("-")
    
    year = int(year_str)
    middle_val = int(middle)
    last_val = int(last)
    hour = int(hour_str)
    minute = int(min_str)
    second = int(sec_str)
    
    # Determine if it's DD-MM or MM-DD format
    # If middle > 12, it must be day (DD-MM format)
    # If last > 12, it must be day (MM-DD format)
    # If both <= 12, prefer DD-MM format (as per the example: 2025-19-08 is day 19, month 08)
    
    if middle_val > 12:
        # middle is day, last is month (DD-MM format)
        day = middle_val
        month = last_val
    elif last_val > 12:
        # middle is month, last is day (MM-DD format)
        month = middle_val
        day = last_val
    else:
        # Ambiguous case - based on the user's example, prefer DD-MM format
        # (2025-19-08 means day 19, month 08)
        day = middle_val
        month = last_val
    
    return datetime(year, month, day, hour, minute, second)

def parse_times_from_filename(name: str):
    """
    Returns (pixel_id:int, start_dt:datetime, end_dt:datetime), all naive (local).
    Raises ValueError if format invalid or end<=start.
    """
    m = _DATE_RE.match(name)
    if not m:
        raise ValueError(f"Unsupported filename format: {name!r}")
    start = _smart_parse_date(m.group("start"))
    end = _smart_parse_date(m.group("end"))
    if end <= start:
        raise ValueError(f"End <= start in filename: {name!r}")
    return int(m.group("pixel")), start, end