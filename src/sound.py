import pygame


class Sound:

    def __init__(self, path):
        self.path = path
        try:
            self.sound = pygame.mixer.Sound(path)
            self.sound_loaded = True
        except FileNotFoundError:
            print(f"Warning: Sound file '{path}' not found. Sound will be disabled.")
            self.sound_loaded = False

    def play(self):
        if hasattr(self, 'sound_loaded') and self.sound_loaded:
            pygame.mixer.Sound.play(self.sound)
