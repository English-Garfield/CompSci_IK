# created by Isaac Korda #
#       26/04/2024       #

import pygame
from src.const import *
from board import Board
from dragger import Drag
from config import Config
from square import Square


class Game:

    def __init__(self):
        self.board = Board()
        self.hovered_sqr = None
        self.drag = Drag()
        self.next_player = 'white'
        self.config = Config()

    # Show methods

    def showBackground(self, surface):
        theme = self.config.theme

        for row in range(ROWS):
            for colum in range(COLUMNS):
                # colour
                colour = theme.bg.light if (row + colum) % 2 == 0 else theme.bg.dark
                # rect
                rect = (colum * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                # blit
                pygame.draw.rect(surface, colour, rect)

                # row coordinates
                if colum == 0:
                    # colour
                    colour = theme.bg.dark if row % 2 == 0 else theme.bg.light
                    # Label
                    label = self.config.font.render(str(ROWS - row), 1, colour)
                    label_pos = (5, 5 + row * SQUARE_SIZE)
                    # Blit
                    surface.blit(label, label_pos)

                # Column coordinates
                if row == 7:
                    # colour
                    colour = theme.bg.dark if colum % 2 == 0 else theme.bg.light
                    # Label
                    label = self.config.font.render(Square.get_ALPHACOLS(colum), 1, colour)
                    label_pos = (colum * SQUARE_SIZE * SQUARE_SIZE - 20, HEIGHT - 20)

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
        theme = self.config.theme

        if self.drag.dragging:
            piece = self.drag.piece

            # Loop all valid moves
            for move in piece.moves:
                # colour
                colour = theme.moves.light if (move.final.row + move.final.colum) % 2 == 0 else theme.moves.dark

                # rectangle
                rectangle = (move.final.colum * SQUARE_SIZE, move.final.row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)

                # Blit
                pygame.draw.rect(surface, colour, rectangle)

    def show_last_move(self, surface):
        theme = self.config.theme

        if self.board.last_move:
            initial = self.board.last_move.initial
            final = self.board.last_move.final

            for pos in [initial, final]:
                # colour
                colour = theme.trace.light if (pos.row * pos.colum) % 2 == 0 else theme.trace.dark
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
            pygame.draw.rect(surface, colour, rect, width=3)

    # other methods
    def nextTurn(self):
        self.next_player = 'white' if self.nextTurn == 'black' else 'black'

    def set_hover(self, row, colum):
        self.hovered_sqr = self.board.squares[row][colum]

    def changeTheme(self):
        self.config.change_theme()

    def soundEffect(self, capture=False):
        if capture:
            self.config.capturedSound.play()
        else:
            self.config.moveSound.play()
