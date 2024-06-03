# created by Isaac Korda #
#       26/04/2024       #

from src.const import *
from square import Square
from piece import *
from move import Move


class Board:

    def __init__(self):
        self.squares = [[0, 0, 0, 0, 0, 0, 0, 0, 0] for colum in range(COLUMNS)]
        self.last_move = None
        self.create_()
        self._add_pieces("white")
        self._add_pieces("black")

    def move(self, piece, move):
        initial = move.initial
        final = move.final

        # console board move update
        self.squares[initial.row][initial.colum].piece = None
        self.squares[final.row][final.colum].piece = piece

        # move
        piece.moved = True

        # clear valid moves
        piece.clear_moves()

        # set last move
        self.last_move = move

    def valid_move(self, piece, move):
        return move in piece.moves

    def calc_moves(self, piece, row, colum):
        # Calculate all possible moves (valid) of a specific piece on a specific position
        def pawn_moves():
            # Steps
            steps = 1 if piece.moved else 2

            # vertical columns
            start = row + piece.dir
            end = row + (piece.dir * (1 + steps))
            for possible_move_row in range(start, end, piece.dir):
                if Square.in_range(possible_move_row):
                    if self.squares[possible_move_row][colum].isEmpty():
                        # create initial final move squares
                        initial = Square(row, colum)
                        final = Square(possible_move_row, colum)
                        # Create a new move
                        move = Move(initial, final)
                        piece.add_move(move)
                    # Blocked
                    else:
                        break
                # not in range
                else:
                    break

            # diagonal columns
            possible_move_row = row + piece.dir
            possible_move_colum = [colum - 1, colum + 1]
            for possible_move_colum in possible_move_colum:
                if Square.in_range(possible_move_row, possible_move_colum):
                    if self.squares[possible_move_row][possible_move_colum].has_rival_piece(piece.colour):
                        # Create initial and final move squares
                        initial = Square(row, colum)
                        final = Square(possible_move_row, possible_move_colum)
                        # Create a new move
                        move = Move(initial, final)
                        # Append new move
                        piece.add_move(move)

        def knight_moves():
            # 8 possible moves
            possible_moves = [
                (row - 2, colum + 1),
                (row - 1, colum + 2),
                (row + 1, colum + 2),
                (row + 2, colum + 1),
                (row + 2, colum - 1),
                (row + 1, colum - 2),
                (row - 1, colum - 2),
                (row - 2, colum - 1),

            ]

            for possible_moves in possible_moves:
                possible_moves_row, possible_moves_colum = possible_moves
                if Square.in_range(possible_moves_row, possible_moves_colum):
                    if self.squares[possible_moves_row][possible_moves_colum].is_empty_or_rival(piece.colour):
                        # create squares of the move
                        initial = Square(row, colum)
                        final = Square(possible_moves_row, possible_moves_colum)  # piece = piece

                        # create new move
                        move = Move(initial, final)
                        # append new valid move
                        piece.add_move(move)

        def king_moves():
            adjacent = [
                (row - 1, colum + 0),  # Up
                (row - 1, colum + 1),  # up right
                (row - 0, colum + 1),  # right
                (row + 1, colum + 1),  # down right
                (row + 1, colum + 0),  # down
                (row + 1, colum - 1),  # down left
                (row + 0, colum - 1),  # left
                (row - 1, colum - 1),  # up left
            ]

            # normal moves
            for possible_move in adjacent:
                possible_move_row, possible_move_colum = possible_move

                if Square.in_range(possible_move_row, possible_move_colum):
                    if self.squares[possible_move_row][possible_move_colum].is_empty_or_rival(piece.colour):
                        # create squares of the move
                        initial = Square(row, colum)
                        final = Square(possible_move_row, possible_move_colum)  # piece = piece
                        # create new move
                        move = Move(initial, final)
                        # append new valid move
                        piece.add_move(move)

        # castling moves

        # queen

        # king

        def straight_line_moves(increments):
            for increments in increments:
                row_increment, colum_increments = increments
                possible_move_row = row + row_increment
                possible_move_colum = colum_increments + colum

                while True:
                    if Square.in_range(possible_move_row, possible_move_colum):
                        # create squares of the possible new move
                        initial = Square(row, colum)
                        final = Square(possible_move_row, possible_move_colum)

                        # possible new move
                        move = Move(initial, final)

                        # Empty = continue looping
                        if self.squares[possible_move_row][possible_move_colum].isEmpty():
                            # append a new move
                            piece.add_move(move)

                        # has rival piece = add move + break
                        if self.squares[possible_move_row][possible_move_colum].has_rival_piece(piece.colour):
                            # append a new move
                            piece.add_move(move)
                            break

                        # has team piece
                        if self.squares[possible_move_row][possible_move_colum].has_team_piece(piece.colour):
                            break

                    # not in range
                    else:
                        break

                    # incrementing the increments
                    possible_move_row = possible_move_row + row_increment
                    possible_move_colum = possible_move_colum + colum_increments

        if isinstance(piece, Pawn):
            pawn_moves()

        elif isinstance(piece, Knight):
            knight_moves()

        elif isinstance(piece, Rook):
            straight_line_moves([
                (-1, 0),  # up
                (0, 1),  # right
                (1, 0),  # down
                (-1, 0)  # left
            ])

        elif isinstance(piece, Bishop):
            straight_line_moves([
                (-1, +1),  # up right
                (-1, -1),  # up left
                (1, 1),  # down right
                (1, -1),  # down left
            ])

        elif isinstance(piece, Queen):
            straight_line_moves([
                (-1, +1),  # up right
                (-1, -1),  # up left
                (1, 1),  # down right
                (1, -1),  # down left

                (-1, 0),  # up
                (0, 1),  # left
                (1, 0),  # down
                (-1, 0)  # left
            ])

        elif isinstance(piece, King):
            king_moves()

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
