# created by Isaac Korda #
#       19/04/2024       #

import sys

# Imports
import pygame

from src.const import *
from src.game import Game
from square import Square
from move import Move


# Main
class Main:

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Chess")
        self.game = Game()

    def mainloop(self):

        game = self.game
        screen = self.screen
        drag = self.game.drag
        board = self.game.board

        while True:
            # Show methods
            game.showBackground(screen)
            game.show_moves(screen)
            game.show_pieces(screen)

            if drag.dragging:
                drag.update_blit(screen)

            for event in pygame.event.get():

                # Click
                if event.type == pygame.MOUSEBUTTONDOWN:
                    drag.updateMouse(event.pos)

                    clicked_row = drag.mouse_y // SQUARE_SIZE
                    clicked_colum = drag.mouse_x // SQUARE_SIZE

                    # if clicked square has a piece ?
                    if board.squares[clicked_row][clicked_colum].has_piece():
                        piece = board.squares[clicked_row][clicked_colum].piece
                        board.calc_moves(piece, clicked_row, clicked_colum)
                        drag.save_initial(event.pos)
                        drag.drag_piece(piece)

                        # show methods
                        game.showBackground(screen)
                        game.show_moves(screen)
                        game.show_pieces(screen)

                # Mouse Motion
                elif event.type == pygame.MOUSEMOTION:
                    if drag.dragging:
                        drag.updateMouse(event.pos)

                        # Show methods
                        game.showBackground(screen)
                        game.show_moves(screen)
                        drag.update_blit(screen)
                        game.show_pieces(screen)

                # mouse release
                elif event.type == pygame.MOUSEBUTTONUP:

                    if drag.dragging:
                        drag.updateMouse(event.pos)

                    released_row = drag.mouse_y // SQUARE_SIZE
                    released_colum = drag.mouse_x // SQUARE_SIZE

                    # create possible moves
                    initial = Square(drag.initial_row, drag.initial_colum)
                    final = Square(released_row, released_colum)
                    move = Move(initial, final)

                    # valid move ?
                    if board.valid_move(drag.piece, move):
                        board.move(drag.piece, move)
                        # show methods
                        game.showBackground(screen)
                        game.show_pieces(screen)

                    drag.undrag_piece()

                # quit application
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            pygame.display.update()


main = Main()
main.mainloop()
