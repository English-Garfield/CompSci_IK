# Imports
import sys
import pygame
import FEN
from sys import maxsize

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

# main loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
