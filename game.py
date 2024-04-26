# created by Isaac Korda #
#       26/04/2024       #

import pygame
from const import *


class Game:

    def __init__(self):
        pass

    # Show methods

    def showBackground(self, surface):
        for row in range(ROWS):
            for colum in range(COLUMNS):
                if (row + colum) % 2 == 0:
                    colour = (234, 234, 200)  # light green

                else:
                    colour = (119, 154, 88)  # Dark green

                rect = (colum * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)

                pygame.draw.rect(surface, colour, rect)
