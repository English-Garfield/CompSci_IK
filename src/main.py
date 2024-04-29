# created by Isaac Korda #
#       19/04/2024       #
import sys

# Imports
import pygame

from src.const import *
from src.game import Game


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
            self.game.showBackground(screen)
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
                        drag.save_initial(event.pos)
                        drag.drag_piece(piece)

                # Mouse Motion
                elif event.type == pygame.MOUSEMOTION:
                    if drag.dragging:
                        drag.updateMouse(event.pos)
                        drag.update_blit(screen)

                # mouse release
                elif event.type == pygame.MOUSEBUTTONUP:
                    drag.undrag_piece()

                # quit application
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            pygame.display.update()


main = Main()
main.mainloop()
