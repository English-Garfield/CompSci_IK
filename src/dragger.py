# created by Isaac Korda #
#       26/04/2024       #

import pygame

from const import *


class Drag:

    def __init__(self):
        self.piece = None
        self.dragging = False
        self.mouse_x = 0
        self.mouse_y = 0
        self.initial_row = 0
        self.initial_colum = 0

    # blit method

    def update_blit(self, surface):
        # texture
        self.piece.set_texture(size=128)
        texture = self.piece.texture
        # img
        img = pygame.image.load(texture)
        # rect
        img_center = (self.mouse_x, self.mouse_y)
        self.piece.texture_rect = img.get_rect(center=img_center)
        # blit
        surface.blit(img, self.piece.texture_rect)

    # other methods

    def updateMouse(self, pos):
        self.mouse_x, self.mouse_y = pos  # (xcor, ycor)

    def save_initial(self, pos):
        self.initial_row = pos[1] // SQUARE_SIZE
        self.initial_colum = pos[0] // SQUARE_SIZE

    def drag_piece(self, piece):
        self.piece = piece
        self.dragging = True

    def undrag_piece(self):
        self.piece = None
        self.dragging = False
