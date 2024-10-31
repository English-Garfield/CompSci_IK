"""holds the main loop of the game"""

import pygame
import sys
from const import *
from game import Game
from square import Square
from move import Move
from piece import Piece

# Initialize Pygame
pygame.init()
# Set up display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Main Screen")

# Button positions
start_button_rect = pygame.Rect((WIDTH // 2 - BUTTON_WIDTH // 2, 200), (BUTTON_WIDTH, BUTTON_HEIGHT))
quit_button_rect = pygame.Rect((WIDTH // 2 - BUTTON_WIDTH // 2, 350), (BUTTON_WIDTH, BUTTON_HEIGHT))


class Main:
    def __init__(self):
        self.checkmate_displayed = None
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Chess')
        self.game = Game()
        self.running = True
        self.in_main_menu = True

    def main_screen(self):
        while self.in_main_menu:
            self.screen.fill(YELLOW)
            pygame.draw.rect(self.screen, GREEN, start_button_rect)
            pygame.draw.rect(self.screen, SALMON, quit_button_rect)

            start_text = FONT.render('Start Game', True, WHITE)
            quit_text = FONT.render('Quit', True, WHITE)

            start_text_rect = start_text.get_rect(center=start_button_rect.center)
            quit_text_rect = quit_text.get_rect(center=quit_button_rect.center)

            self.screen.blit(start_text, start_text_rect)
            self.screen.blit(quit_text, quit_text_rect)

            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.in_main_menu = False
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if start_button_rect.collidepoint(event.pos):
                        self.in_main_menu = False
                    elif quit_button_rect.collidepoint(event.pos):
                        pygame.quit()
                        sys.exit()

    def mainloop(self):
        self.main_screen()

        while self.running:
            self.screen.fill(DARK_GREEN)
            self.game.show_bg(self.screen)
            self.game.show_last_move(self.screen)
            self.game.show_moves(self.screen)
            self.game.show_pieces(self.screen)
            self.game.show_hover(self.screen)

            if self.game.dragger.dragging:
                self.game.dragger.update_blit(self.screen)

            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.game.dragger.update_mouse(event.pos)
                    clicked_row = self.game.dragger.mouseY // SQSIZE
                    clicked_col = self.game.dragger.mouseX // SQSIZE
                    if self.game.board.squares[clicked_row][clicked_col].has_piece():
                        piece = self.game.board.squares[clicked_row][clicked_col].piece
                        if piece.color == self.game.next_player:
                            self.game.board.calc_moves(piece, clicked_row, clicked_col, bool=True)
                            self.game.dragger.save_initial(event.pos)
                            self.game.dragger.drag_piece(piece)
                            self.game.show_bg(self.screen)
                            self.game.show_last_move(self.screen)
                            self.game.show_moves(self.screen)
                            self.game.show_pieces(self.screen)

                elif event.type == pygame.MOUSEMOTION:
                    motion_row = event.pos[1] // SQSIZE
                    motion_col = event.pos[0] // SQSIZE
                    self.game.set_hover(motion_row, motion_col)
                    if self.game.dragger.dragging:
                        self.game.dragger.update_mouse(event.pos)
                        self.game.show_bg(self.screen)
                        self.game.show_last_move(self.screen)
                        self.game.show_moves(self.screen)
                        self.game.show_pieces(self.screen)
                        self.game.show_hover(self.screen)
                        self.game.dragger.update_blit(self.screen)

                elif event.type == pygame.MOUSEBUTTONUP:
                    if self.game.dragger.dragging:
                        self.game.dragger.update_mouse(event.pos)
                        released_row = self.game.dragger.mouseY // SQSIZE
                        released_col = self.game.dragger.mouseX // SQSIZE
                        initial = Square(self.game.dragger.initial_row, self.game.dragger.initial_col)
                        final = Square(released_row, released_col)
                        move = Move(initial, final)

                        if self.game.board.valid_move(self.game.dragger.piece, move):
                            captured = self.game.board.squares[released_row][released_col].has_piece()
                            self.game.board.move(self.game.dragger.piece, move)
                            self.game.board.set_true_en_passant(self.game.dragger.piece)
                            self.game.play_sound(captured)
                            self.game.show_bg(self.screen)
                            self.game.show_last_move(self.screen)
                            self.game.show_pieces(self.screen)
                            self.game.next_turn()

                    self.game.dragger.undrag_piece()

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_t:
                        self.game.change_theme()

                    if event.key == pygame.K_r:
                        self.game.reset()
                        self.game = Game()

                pygame.display.update()


main = Main()
main.mainloop()
