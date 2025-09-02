#!/usr/bin/env python3
"""
Migration script to convert PyQt5 to PySide6.
"""

import os
import re
from pathlib import Path

# Mapping of PyQt5 to PySide6 imports
IMPORT_MAPPINGS = {
    'from PyQt5.QtWidgets': 'from PySide6.QtWidgets',
    'from PyQt5.QtCore': 'from PySide6.QtCore',
    'from PyQt5.QtGui': 'from PySide6.QtGui',
    'from PyQt5': 'from PySide6',
    'import PyQt5': 'import PySide6',
    'PyQt5.': 'PySide6.',
    
    # Signal/slot changes
    'pyqtSignal': 'Signal',
    'pyqtSlot': 'Slot',
    
    # exec_ -> exec (Python 3 compatible)
    '.exec_()': '.exec()',
    
    # QString is gone in PySide6 (uses native Python strings)
    'QString': 'str',
}

# Additional method/attribute changes
METHOD_CHANGES = [
    # QAction moved from QtWidgets to QtGui in Qt6
    (r'from PySide6\.QtWidgets import (.*?)QAction', r'from PySide6.QtGui import QAction\nfrom PySide6.QtWidgets import \1'),
    
    # Some enum changes
    (r'Qt\.MidButton', r'Qt.MiddleButton'),
    
    # High DPI changes
    (r'Qt\.AA_EnableHighDpiScaling', r'Qt.AA_UseHighDpiPixmaps'),  # Simplified for Qt6
]

def migrate_file(filepath):
    """Migrate a single Python file from PyQt5 to PySide6."""
    print(f"Migrating {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Apply import mappings
    for old, new in IMPORT_MAPPINGS.items():
        content = content.replace(old, new)
    
    # Apply method/attribute changes
    for pattern, replacement in METHOD_CHANGES:
        content = re.sub(pattern, replacement, content)
    
    # Fix Signal imports if needed
    if 'Signal' in content and 'from PySide6.QtCore import' in content:
        # Make sure Signal is imported
        if 'from PySide6.QtCore import Signal' not in content:
            content = re.sub(
                r'from PySide6\.QtCore import ([^\\n]+)',
                r'from PySide6.QtCore import \1, Signal',
                content,
                count=1
            )
    
    # Write back if changed
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ Migrated {filepath}")
        return True
    else:
        print(f"  - No changes needed for {filepath}")
        return False

def main():
    """Main migration function."""
    print("Starting PyQt5 to PySide6 migration...")
    
    # Find all Python files in spectrogram_gui
    gui_path = Path('/workspace/spectrogram_gui')
    python_files = list(gui_path.rglob('*.py'))
    
    migrated_count = 0
    for filepath in python_files:
        if migrate_file(filepath):
            migrated_count += 1
    
    print(f"\nMigration complete! Migrated {migrated_count}/{len(python_files)} files.")
    
    # Update requirements.txt
    req_file = gui_path / 'requirements.txt'
    if req_file.exists():
        print("\nUpdating requirements.txt...")
        with open(req_file, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            if 'PyQt5' in line:
                new_lines.append('PySide6>=6.5.0\n')
            elif 'qdarkstyle' in line:
                # qdarkstyle may not support PySide6 directly, we'll handle this
                new_lines.append('# qdarkstyle>=3.2  # Replaced with custom theme\n')
            else:
                new_lines.append(line)
        
        with open(req_file, 'w') as f:
            f.writelines(new_lines)
        print("  ✓ Updated requirements.txt")

if __name__ == '__main__':
    main()