# Time Axis & CSV Fix + UI Enhancements

## PR Title
`fix(time,ui): axis from filename start (local-naive) + correct CSV events + sort menu`

## Summary of Changes

This PR fixes the time axis display bug and adds small UI enhancements as requested. All changes are minimal, surgical fixes that maintain backward compatibility and don't alter the core DSP/playback logic.

## 🐞 Bug Fixes

### 1. **Fixed Time Axis Display** ✅
- **Problem**: Time axis showed wrong origin (e.g., "14:00:00") instead of actual filename start time
- **Root Cause**: Using `datetime.fromtimestamp()` fallback which introduced UTC/epoch offsets
- **Solution**: 
  - Created centralized time parser in `spectrogram_gui/utils/time_parse.py`
  - Parser extracts BOTH start and end times from filename
  - Supports both DD-MM and MM-DD date formats intelligently
  - Axis now displays full datetime format (YYYY-MM-DD HH:MM:SS) to avoid confusion
  - All times treated as local-naive (no timezone conversions)

### 2. **Fixed CSV Export Times** ✅
- **Problem**: Event times in CSV didn't match absolute times from filename
- **Solution**: 
  - Removed microsecond precision that was causing format issues
  - CSV now uses clean `YYYY-MM-DD HH:MM:SS` format
  - Times computed correctly as `file_start_dt + timedelta(seconds=offset)`

## 🎨 UI Enhancements

### 3. **Right-Click Sort Menu** ✅
- Added comprehensive sorting options for file list:
  - Sort by: Name, Start Time, End Time, Duration
  - Ascending/Descending toggle
  - Settings persist across sessions using QSettings
- Enhanced multi-select support (Shift/Ctrl)
- Context menu actions:
  - Mark Event
  - Show Only Selected (filters view to selected files)
  - Remove from View (non-destructive)

### 4. **Light Theme Enhancements** ✅
- Created `spectrogram_gui/styles/app_light.qss`
- Subtle improvements:
  - Better hover states for lists and buttons
  - Enhanced selection highlighting
  - Cleaner tooltips and menus
  - Improved scrollbar visibility
- Non-breaking: loads only if file exists, doesn't conflict with existing themes

## 📦 Files Changed

### New Files:
- `spectrogram_gui/utils/time_parse.py` - Centralized time parsing
- `spectrogram_gui/styles/app_light.qss` - Light theme enhancements
- `tests/test_time_parse.py` - Comprehensive tests

### Modified Files:
- `spectrogram_gui/gui/main_window.py` - Updated file loading, added sort menu
- `spectrogram_gui/gui/spectrogram_canvas.py` - Fixed axis display
- `spectrogram_gui/gui/event_annotator.py` - Fixed CSV format
- `spectrogram_gui/main.py` - Added light theme loading

## 🧪 Testing

### Automated Tests
All tests pass:
```
✓ test_dd_mm_variant
✓ test_iso_variant
✓ test_ambiguous_date
✓ test_invalid_format
✓ test_end_before_start
✓ test_with_extra_whitespace
✓ test_midnight_crossing
```

### Manual QA Checklist

1. **Time Axis Display** ✅
   - Load: `pixel - 211 - 2025-19-08 10-19-59 - 2025-19-08 10-28-55`
   - Expected: Left axis = `2025-08-19 10:19:59`, Right = `2025-08-19 10:28:55`
   - Duration: ~8:56
   - NO "14:00" offset

2. **CSV Export** ✅
   - Mark event 60s after start
   - CSV time should be `2025-08-19 10:20:59`

3. **ISO Format** ✅
   - Load: `pixel - 2221 - 2025-05-29 10-46-00 - 2025-05-29 11-08-00`
   - Axis shows `10:46:00` to `11:08:00`

4. **Sorting** ✅
   - Right-click file list → Sort menu works
   - Settings persist after restart

## 🛡️ Safety Guarantees

- ✅ Python 3.8 compatible (no walrus operator, no match/case)
- ✅ No new dependencies added
- ✅ All datetime handling is local-naive (no UTC/tz conversions)
- ✅ Backward compatible - old files without proper naming still work
- ✅ No changes to DSP/playback logic
- ✅ Minimal diffs (~200 LOC total changes)

## Search Points Fixed

### A) Time Parsing
- Created centralized parser
- Replaced all ad-hoc parsing

### B) Axis Mapping
- Fixed to use `file_start_dt + timedelta(seconds=x)`
- Removed epoch/UTC conversions

### C) CSV I/O
- Fixed to use local-naive format
- Consistent YYYY-MM-DD HH:MM:SS format

## Notes

The implementation follows the exact specifications provided, with surgical fixes that don't refactor the architecture. The UI enhancements are optional and non-breaking, adding value without risk.