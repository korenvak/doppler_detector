# Final Fix: File Extension Support & Correct Date Parsing

## Root Cause Identified
The filename parser was **failing silently** because the regex pattern didn't account for file extensions (.flac, .wav, etc). This caused the code to fall back to using `datetime.now()`, which explained why you were seeing incorrect times like "12:42" that had nothing to do with your actual file.

## Complete Fix Applied

### 1. **Fixed Regex Pattern to Support File Extensions**
```python
# BEFORE: Pattern ended with $ (end of string)
(?P<end>\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}-\d{2}-\d{2})\s*$

# AFTER: Added optional file extension pattern
(?P<end>\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}-\d{2}-\d{2})
(?:\.\w+)?  # Optional file extension
\s*$
```

### 2. **Fixed Date Format Logic (DD-MM vs MM-DD)**
The parser now correctly identifies:
- `2025-19-08` → August 19, 2025 (day=19, month=08)
- `2025-05-29` → May 29, 2025 (month=05, day=29)

### 3. **Removed Misleading datetime.now() Fallback**
Changed from using current time to a fixed reference date (2000-01-01) when filename can't be parsed, making it obvious when timestamps aren't real.

### 4. **Time Axis Shows Only HH:MM:SS**
- Axis displays: `12:47:45` instead of `2025-08-19 12:47:45`
- CSV still exports full datetime: `2025-08-19 12:47:45`

## Test Results ✅
```
✓ All original tests pass
✓ test_user_specific_example (your exact filename)
✓ test_with_file_extension (.flac, .wav, .mp3)
```

## Expected Behavior Now

For your file: `pixel - 1567 - 2025-19-08 12-47-45 - 2025-19-08 13-09-53.flac`

### Time Axis:
- **Start**: `12:47:45` (not 12:42 or current time!)
- **End**: `13:09:53`
- Shows only time, no date

### CSV Export:
```
Start                End
2025-08-19 12:47:45  2025-08-19 13:09:53
```
- Full date + time
- Correct date: August 19, 2025

### Event Marking:
- Works correctly with proper timestamps
- Events saved at correct absolute times

## Files Changed
1. `/workspace/spectrogram_gui/utils/time_parse.py` - Added file extension support
2. `/workspace/spectrogram_gui/gui/main_window.py` - Better fallback handling
3. `/workspace/spectrogram_gui/gui/spectrogram_canvas.py` - Time-only display format

## Key Insight
The issue wasn't with the date parsing logic itself, but with the **regex pattern not matching filenames with extensions**, causing a silent fallback to incorrect timestamps. This has been completely fixed.