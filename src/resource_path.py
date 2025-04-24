import os
import sys

def resource_path(relative_path):
    """
    Get the absolute path to a resource, works for development and for PyInstaller
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # If not running as a PyInstaller executable, use the script's directory
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)