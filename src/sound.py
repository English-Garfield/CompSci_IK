import pygame
import sys
from resource_path import resource_path


class Sound:

    def __init__(self, path):
        self.path = path
        self.sound_loaded = False
        
        try:
            # Check if running in a browser environment
            in_browser = hasattr(sys, 'javascript_is_available') and sys.javascript_is_available
            
            # Initialize mixer if not already initialized
            if not pygame.mixer.get_init():
                try:
                    pygame.mixer.init()
                except Exception as e:
                    print(f"Warning: Could not initialize sound mixer: {e}")
                    return
            
            # Load the sound
            self.sound = pygame.mixer.Sound(path)
            self.sound_loaded = True
            
            if in_browser:
                print(f"Sound loaded in browser: {path}")
        except FileNotFoundError:
            print(f"Warning: Sound file '{path}' not found. Sound will be disabled.")
        except Exception as e:
            print(f"Warning: Could not load sound '{path}': {e}")

    def play(self):
        if hasattr(self, 'sound_loaded') and self.sound_loaded:
            try:
                pygame.mixer.Sound.play(self.sound)
            except Exception as e:
                print(f"Warning: Could not play sound: {e}")
                # Don't set sound_loaded to False here to allow retrying
