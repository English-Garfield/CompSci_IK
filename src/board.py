# created by Isaac Korda #
#       26/04/2024       #

from src.const import *
from square import Square
from piece import *


class Board:

    def __init__(self):
        self.squares = [[0, 0, 0, 0, 0, 0, 0, 0, 0] for colum in range(COLUMNS)]

        self.create_()
        self._add_pieces("white")
        self._add_pieces("black")

    def create_(self):

        for row in range(ROWS):
            for colum in range(COLUMNS):
                self.squares[row][colum] = Square(row, colum)

    def _add_pieces(self, colour):
        row_pawn, row_other = (6, 7) if colour == "white" else (1, 0)

        # Pawns
        for colum in range(COLUMNS):
            self.squares[row_pawn][colum] = Square(row_pawn, colum, Pawn(colour))

        # Knights
        self.squares[row_other][1] = Square(row_other, 1, Knight(colour))
        self.squares[row_other][6] = Square(row_other, 6, Knight(colour))

        # Bishops
        self.squares[row_other][2] = Square(row_other, 2, Bishop(colour))
        self.squares[row_other][5] = Square(row_other, 6, Bishop(colour))

        # Rooks
        self.squares[row_other][0] = Square(row_other, 0, Rook(colour))
        self.squares[row_other][7] = Square(row_other, 7, Rook(colour))

        # Queen
        self.squares[row_other][3] = Square(row_other, 3, Queen(colour))

        # King
        self.squares[row_other][4] = Square(row_other, 4, King(colour))


b = Board()
b.create_()
