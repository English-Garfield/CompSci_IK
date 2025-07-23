import os
import sys
import pygame

def resource_path(relative_path):
    """
    Get the absolute path to a resource, works for:
    - Development environment
    - PyInstaller executable
    - Web deployment (Pygbag)
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # Check if running in a browser environment (Pygbag)
        if hasattr(sys, 'javascript_is_available') and sys.javascript_is_available:
            # In browser, use relative paths directly
            return relative_path
        else:
            # If not running as a PyInstaller executable, use the script's directory
            base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)