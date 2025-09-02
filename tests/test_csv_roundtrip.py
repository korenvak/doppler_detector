"""
Test CSV I/O round-trip to ensure event times are stable.
"""

import unittest
import tempfile
import os
import csv
from datetime import datetime, timedelta
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCSVRoundTrip(unittest.TestCase):
    """Test that CSV export/import preserves timestamps correctly."""
    
    def setUp(self):
        """Create a temporary CSV file for testing."""
        self.temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.temp_csv.close()
        
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_csv.name):
            os.unlink(self.temp_csv.name)
    
    def test_event_csv_roundtrip(self):
        """Test that events written to CSV can be read back identically."""
        # Create test events with various timestamps
        test_events = [
            {'time': '2025-08-19 12:47:45', 'event': 'Start', 'notes': 'Test start'},
            {'time': '2025-08-19 12:55:30', 'event': 'Middle', 'notes': 'Mid point'},
            {'time': '2025-08-19 13:09:53', 'event': 'End', 'notes': 'Test end'},
        ]
        
        # Write events to CSV
        with open(self.temp_csv.name, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['time', 'event', 'notes'])
            writer.writeheader()
            writer.writerows(test_events)
        
        # Read events back
        with open(self.temp_csv.name, 'r') as f:
            reader = csv.DictReader(f)
            read_events = list(reader)
        
        # Verify all data matches
        self.assertEqual(len(read_events), len(test_events))
        for i, event in enumerate(test_events):
            self.assertEqual(read_events[i]['time'], event['time'])
            self.assertEqual(read_events[i]['event'], event['event'])
            self.assertEqual(read_events[i]['notes'], event['notes'])
    
    def test_timestamp_format_consistency(self):
        """Test that timestamps maintain YYYY-MM-DD HH:MM:SS format."""
        # Create events with different time formats
        test_events = [
            {'time': '2025-08-19 00:00:00', 'event': 'Midnight'},
            {'time': '2025-12-31 23:59:59', 'event': 'Year end'},
            {'time': '2025-01-01 00:00:01', 'event': 'Year start'},
        ]
        
        # Write to CSV
        with open(self.temp_csv.name, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['time', 'event'])
            writer.writeheader()
            writer.writerows(test_events)
        
        # Read back
        with open(self.temp_csv.name, 'r') as f:
            reader = csv.DictReader(f)
            read_events = list(reader)
        
        for i, event in enumerate(test_events):
            read_time = read_events[i]['time']
            # Verify format is maintained
            self.assertEqual(read_time, event['time'])
            # Verify it can be parsed as datetime
            dt = datetime.strptime(read_time, '%Y-%m-%d %H:%M:%S')
            self.assertIsInstance(dt, datetime)
    
    def test_relative_to_absolute_time_conversion(self):
        """Test conversion between relative seconds and absolute timestamps."""
        start_dt = datetime(2025, 8, 19, 12, 47, 45)
        
        # Test various relative times
        test_cases = [
            (0, '2025-08-19 12:47:45'),      # Start
            (60, '2025-08-19 12:48:45'),     # 1 minute later
            (3600, '2025-08-19 13:47:45'),   # 1 hour later
            (1328, '2025-08-19 13:09:53'),   # Specific time from example
        ]
        
        for rel_seconds, expected_str in test_cases:
            absolute_dt = start_dt + timedelta(seconds=rel_seconds)
            result_str = absolute_dt.strftime('%Y-%m-%d %H:%M:%S')
            self.assertEqual(result_str, expected_str)
            
            # Verify round-trip
            parsed_dt = datetime.strptime(result_str, '%Y-%m-%d %H:%M:%S')
            delta = (parsed_dt - start_dt).total_seconds()
            self.assertAlmostEqual(delta, rel_seconds, places=1)


if __name__ == '__main__':
    unittest.main()