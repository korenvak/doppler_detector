# app/utils/time_parse.py
import re
from datetime import datetime

_DATE_RE = re.compile(
    r"""^
        \s*pixel\s*-\s*(?P<pixel>\d+)\s*-\s*
        (?P<start>\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}-\d{2}-\d{2})\s*-\s*
        (?P<end>\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}-\d{2}-\d{2})\s*$
    """, re.X
)

def _smart_parse_date(s: str) -> datetime:
    """
    Accepts:
      - YYYY-DD-MM HH-MM-SS  (e.g., 2025-19-08 10-19-59)
      - YYYY-MM-DD HH-MM-SS  (e.g., 2025-05-29 10-46-00)
    Returns naive datetime (local clock, no tz).
    """
    date_part, time_part = s.split()
    y, m_or_d, d_or_m = date_part.split("-")
    h, M, S = time_part.split("-")

    a, b = int(m_or_d), int(d_or_m)
    fmt_iso = "%Y-%m-%d %H-%M-%S"
    fmt_ddm = "%Y-%d-%m %H-%M-%S"

    if a > 12 and b <= 12:
        return datetime.strptime(s, fmt_ddm)
    if b > 12 and a <= 12:
        return datetime.strptime(s, fmt_iso)

    # Ambiguous (both <=12): try ISO first, then DD-MM
    try:
        return datetime.strptime(s, fmt_iso)
    except ValueError:
        return datetime.strptime(s, fmt_ddm)

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