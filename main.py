"""
Wrapper module for the chess game.
This file serves as the entry point for the web demo.
"""

import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the main module from src
from src.main import Main

# Create and run the game
if __name__ == "__main__":
    main = Main()
    main.mainloop()
    import pygame
    pygame.quit()