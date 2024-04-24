# created by Isaac Korda #
#       19/04/2024       #


# Imports
import sys
import pygame
import FEN

pygame.init()

# Set up window
size = (640, 640)
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Chess game")

# Set up board
board = pygame.Surface((600, 600))
board.fill((255, 206, 158))

# Draw the Board
for x in range(0, 8, 2):
    for y in range(0, 8, 2):
        pygame.draw.rect(board, (210, 180, 140), (x * 75, y * 75, 75, 75))
        pygame.draw.rect(board, (210, 180, 140), ((x + 1) * 75, (y + 1) * 75, 75, 75))

# adding board to screen
screen.blit(board, (20, 20))

pygame.display.flip()


class Piece:
    def __init__(self, colour, x, y, piece_type):
        self.colour = colour
        self.x = x
        self.y = y
        self.type = piece_type

    def draw(self, surface):
        image = pygame.image.load(f"assets/{self.colour}_{self.type}.png")
        surface.blit(image, (self.x * 75 + 10, self.y * 75 + 10))


# Set up pieces
pieces = []
for i in range(8):
    pieces.append(Piece("Black", i, 1, "Pawn"))
    pieces.append(Piece("White", i, 6, "Pawn"))

# Draw the pieces
for piece in pieces:
    piece.draw(board)

# main loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
