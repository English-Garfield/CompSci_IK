class Square:

    def __init__(self, row, colum, piece=None):
        self.row = row
        self.colum = colum
        self.piece = piece

    def __eq__(self, other):
        return self.row == other.row and self.colum == other.colum

    def has_piece(self):
        return self.piece != None

    def isEmpty(self):
        return not self.has_piece()

    def has_team_piece(self, colour):
        return self.has_piece() and self.piece.colour == colour

    def has_rival_piece(self, colour):
        return self.has_piece() and self.piece.colour != colour

    def is_empty_or_rival(self, colour):
        return self.isEmpty() or self.has_rival_piece(colour)

    @staticmethod
    def in_range(*args):
        for argument in args:
            if argument < 0 or argument > 7:
                return False

        return True
