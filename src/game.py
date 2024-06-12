# created by Isaac Korda #
#       26/04/2024       #

import pygame
from src.const import *
from board import Board
from dragger import Drag


class Game:

    def __init__(self):
        self.board = Board()
        self.hovered_sqr = None
        self.drag = Drag()
        self.next_player = 'white'

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

                    # all but dragged piece
                    if piece is not self.drag.piece:
                        piece.set_texture(size=80)
                        img = pygame.image.load(piece.texture)
                        img_center = colum * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2
                        piece.texture_rect = img.get_rect(center=img_center)
                        surface.blit(img, piece.texture_rect)

    def show_moves(self, surface):
        if self.drag.dragging:
            piece = self.drag.piece

            # Loop all valid moves
            for move in piece.moves:
                # colour
                colour = '#C86464' if (move.final.row + move.final.colum) % 2 == 0 else '#C84646'

                # rectangle
                rectangle = (move.final.colum * SQUARE_SIZE, move.final.row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)

                # Blit
                pygame.draw.rect(surface, colour, rectangle)

    def show_last_move(self, surface):
        if self.board.last_move:
            initial = self.board.last_move.initial
            final = self.board.last_move.final

            for pos in [initial, final]:
                # colour
                colour = (244, 247, 116) if (pos.row * pos.colum) % 2 == 0 else (172, 195, 51)
                # rect
                rect = (pos.colum * SQUARE_SIZE, pos.row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                # blit
                pygame.draw.rect(surface, colour, rect)

    def show_hover(self, surface):
        if self.hovered_sqr:
            # colour
            colour = (180, 180, 180)
            # rect
            rect = (self.hovered_sqr.colum * SQUARE_SIZE, self.hovered_sqr.row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            # blit
            pygame.draw.rect(surface, colour, rect)

    # other methods
    def nextTurn(self):
        self.next_player = 'white' if self.nextTurn == 'black' else 'black'

    def set_hover(self, row, colum):
        self.hovered_sqr = self.board.squares[row][colum]

