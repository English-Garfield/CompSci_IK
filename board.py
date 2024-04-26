from const import *
from square import Square


class Board:

    def __init__(self):
        self.squares = [[0, 0, 0, 0, 0, 0, 0, 0, 0] for colum in range(COLUMNS)]

        self.create_()

    def create_(self):

        for row in range(ROWS):
            for colum in range(COLUMNS):
                self.squares[row][colum] = Square(row, colum)

    def _add_pieces(self, colour):
        pass


b = Board()
b.create_()
