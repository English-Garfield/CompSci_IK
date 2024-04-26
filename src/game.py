# created by Isaac Korda #
#       26/04/2024       #

import pygame
from src.const import *
from board import Board


class Game:

    def __init__(self):
        self.board = Board()

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

    def show_pieces(self, surface):
        for row in range(ROWS):
            for colum in range(COLUMNS):
                # Piece?
                if self.board.squares[row][colum].has_piece():
                    piece = self.board.squares[row][colum].piece

                    img = pygame.image.load(piece.texture)
                    img_center = colum * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2
                    piece.texture_rect = img.get_rect(center=img_center)
                    surface.blit(img, piece.texture_rect)
