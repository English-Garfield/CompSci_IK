# created by Isaac Korda #
#       19/04/2024       #
import sys

# Imports
import pygame

from src.const import *
from src.game import Game


# Main
class Main:

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Chess")
        self.game = Game()

    def mainloop(self):

        game = self.game
        screen = self.screen

        while True:
            self.game.showBackground(screen)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            pygame.display.update()


main = Main()
main.mainloop()
