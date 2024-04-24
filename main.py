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

# Rooks
pieces.append(Piece("Black", 0, 0, "Rook"))
pieces.append(Piece("Black", 7, 0, "Rook"))

pieces.append(Piece("White", 0, 7, "Rook"))
pieces.append(Piece("White", 7, 7, "Rook"))

# Knights
pieces.append(Piece("Black", 1, 0, "Knight"))
pieces.append(Piece("Black", 6, 0, "Knight"))

pieces.append(Piece("White", 1, 7, "Knight"))
pieces.append(Piece("White",6, 7, "Knight"))

# Bishops
pieces.append(Piece("Black", 2, 0, "Bishop"))
pieces.append(Piece("Black", 5, 0, "Bishop"))

pieces.append(Piece("White", 2, 7, "Bishop"))
pieces.append(Piece("White",5, 7, "Bishop"))


# Draw the pieces
for piece in pieces:
    piece.draw(board)

# main loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            # get the position of the click
            pos = pygame.mouse.get_pos()

            # convert the position to board coordinates
            x = (pos[0] - 20) // 75
            y = (pos[1] - 20) // 75

            # find the piece at the clicked position
            for piece in pieces:
                if piece.x == x and piece.y == y:
                    # move the piece
                    pos = pygame.mouse.get_pos()
                    x = (pos[0] - 20) // 75
                    y = (pos[1] - 20) // 75
                    piece.x = x
                    piece.y = y

    # redraw the board and pieces
    board.fill((255, 206, 158))
    for x in range(0, 8, 2):
        for y in range(0, 8, 2):
            pygame.draw.rect(board, (210, 180, 140), (x * 75, y * 75, 75, 75))
            pygame.draw.rect(board, (210, 180, 140), ((x + 1) * 75, (y + 1) * 75, 75, 75))

    for piece in pieces:
        piece.draw(board)

    # add the board to the screen
    screen.blit(board, (20, 20))

    # update the display
    pygame.display.update()
