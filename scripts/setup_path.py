#!/usr/bin/env python3
"""
Shared utility to setup Python path for script execution.

This module provides a function to add the project root to sys.path,
allowing scripts to import from the 'src' module regardless of how they are invoked.
"""
import os
import sys


def setup_project_path():
    """
    Add project root to Python path.
    
    This function calculates the project root directory (parent of scripts/)
    and adds it to the beginning of sys.path to ensure local 'src' module
    takes precedence over any system-installed package with the same name.
    
    This function is idempotent - calling it multiple times has no adverse effects.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Only add if not already present to avoid duplicates
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
