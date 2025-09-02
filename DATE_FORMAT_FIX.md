# Date Format Fix Summary

## Issues Fixed

### 1. **Incorrect Date Parsing**
- **Problem**: Files like `pixel - 1567 - 2025-19-08 12-47-45 - 2025-19-08 13-09-53.flac` were incorrectly parsed
- **Root Cause**: The parser was trying to interpret "19" as a month, which is invalid
- **Solution**: Rewrote the date parser to correctly identify DD-MM vs MM-DD format:
  - If middle value > 12, it's DD-MM format (day-month)
  - If last value > 12, it's MM-DD format (month-day)  
  - If both ≤ 12, prefer DD-MM format (based on user's examples)

### 2. **Time Axis Showing Full Date**
- **Problem**: Time axis showed `2025-08-19 12:47:45` instead of just `12:47:45`
- **Solution**: Changed axis format to show only `HH:MM:SS`
- **Impact**: Cleaner, more readable time axis

### 3. **CSV Export Format**
- **Verified**: CSV correctly exports full datetime as `YYYY-MM-DD HH:MM:SS`
- **Example**: For the file above, CSV will show:
  - Start: `2025-08-19 12:47:45`
  - End: `2025-08-19 13:09:53`

## Code Changes

### `/workspace/spectrogram_gui/utils/time_parse.py`
```python
# Simplified and corrected date parsing logic
if middle_val > 12:
    # middle is day, last is month (DD-MM format)
    day = middle_val
    month = last_val
elif last_val > 12:
    # middle is month, last is day (MM-DD format)
    month = middle_val
    day = last_val
else:
    # Ambiguous - prefer DD-MM based on user examples
    day = middle_val
    month = last_val
```

### `/workspace/spectrogram_gui/gui/spectrogram_canvas.py`
```python
# Time axis now shows only HH:MM:SS
out.append(t.strftime("%H:%M:%S"))  # Changed from "%Y-%m-%d %H:%M:%S"
```

## Test Results

✅ All tests pass including new test for user's specific example:

```
✓ test_dd_mm_variant
✓ test_iso_variant
✓ test_ambiguous_date
✓ test_invalid_format
✓ test_end_before_start
✓ test_with_extra_whitespace
✓ test_midnight_crossing
✓ test_user_specific_example
```

## Expected Behavior

For file: `pixel - 1567 - 2025-19-08 12-47-45 - 2025-19-08 13-09-53.flac`

**Time Axis Display:**
- Left edge: `12:47:45`
- Right edge: `13:09:53`
- Hover shows: `HH:MM:SS` format

**CSV Export:**
- Start: `2025-08-19 12:47:45` (August 19, 2025)
- End: `2025-08-19 13:09:53` (August 19, 2025)

**Event Marking:**
- Works correctly with proper timestamps
- Events placed at correct absolute times