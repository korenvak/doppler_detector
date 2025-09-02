#!/usr/bin/env python3
"""
Fix remaining PySide6 migration issues.
"""

import os
import re
from pathlib import Path

def fix_file(filepath):
    """Fix any remaining issues in a Python file."""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Fix any remaining Signal typos
    content = re.sub(r'\bSig\b(?!nal)', 'Signal', content)
    content = re.sub(r'\bSignalnal\b', 'Signal', content)
    content = re.sub(r'\bSignalnnal\b', 'Signal', content)
    
    # Fix QRunnable typos
    content = re.sub(r'\bQRu\b', 'QRunnable', content)
    content = re.sub(r'\bQRunnnable\b', 'QRunnable', content)
    
    # Ensure QAction and QShortcut are imported from QtGui
    if 'from PySide6.QtWidgets import' in content and ('QAction' in content or 'QShortcut' in content):
        # Check if these are incorrectly imported from QtWidgets
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'from PySide6.QtWidgets import' in line:
                # Check if QAction or QShortcut are in this import
                if 'QAction' in line or 'QShortcut' in line:
                    # Remove them from QtWidgets import
                    line = re.sub(r',\s*QAction', '', line)
                    line = re.sub(r'QAction\s*,', '', line)
                    line = re.sub(r',\s*QShortcut', '', line)
                    line = re.sub(r'QShortcut\s*,', '', line)
                    lines[i] = line
                    
                    # Add to QtGui import if not already there
                    gui_import_found = False
                    for j, check_line in enumerate(lines):
                        if 'from PySide6.QtGui import' in check_line:
                            if 'QAction' not in check_line:
                                lines[j] = check_line.rstrip() + ', QAction'
                            if 'QShortcut' not in check_line:
                                lines[j] = lines[j].rstrip() + ', QShortcut'
                            gui_import_found = True
                            break
                    
                    if not gui_import_found:
                        # Add QtGui import after QtWidgets
                        lines.insert(i+1, 'from PySide6.QtGui import QAction, QShortcut')
        
        content = '\n'.join(lines)
    
    # Write back if changed
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed: {filepath}")
        return True
    return False

def main():
    """Main function."""
    print("Fixing remaining PySide6 migration issues...")
    
    gui_path = Path('/workspace/spectrogram_gui')
    python_files = list(gui_path.rglob('*.py'))
    
    fixed_count = 0
    for filepath in python_files:
        if fix_file(filepath):
            fixed_count += 1
    
    print(f"Fixed {fixed_count} files.")

if __name__ == '__main__':
    main()