# created by Isaac Korda #
#       26/04/2024       #

import pygame
from const import *


class Drag:
    def __init__(self):
        self.mouse_x = 0
        self.mouse_y = 0
        self.initial_row = 0
        self.initial_colum = 0

    def updateMouse(self, pos):
        self.mouse_x, self.mouse_y = pos # (mouse_x, mouse_y)

    def save_initial(self, pos):
        self.initial_row = pos[1] // SQUARE_SIZE
        self.initial_colum = pos[0] // SQUARE_SIZE

