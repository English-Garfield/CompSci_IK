import numpy as np
import math
import random as rdm

from chess import pgn
from chess import board
from board import Board
from tqdm import tqdm
from const import *

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam

class engine:

    def __init__(self):
        self.file = "../assets/ChessData"

    def laod_pgn(self, path):
        games = []
        with open(path, 'r') as pgn_file:
            while True:
                game = pgn.read_game(pgn_file)
                if game is None:
                    break
                games.append(game)
        return games

        LIMIT_OF_FILES = min(len(files), 24)
        games = []
        i = 1
        for file in tqdm(files):
            games.extend(self.laod_pgn(file))
            if i>= LIMIT_OF_FILES:
                break
            i+=1

    def boardToMatrix(self,board: Board):
        matrix = np.zeros((8, 8, 12))
        piece_map = board.piece_map()
        for square, piece in piece_map.items():
            ROW, COLS = divmod(square, 8)
            piece_type = piece.piece_type - 1
            piece_colour = 0 if piece.color else 6
            matrix[ROW, COLS, piece_type + piece_colour] = 1
        return matrix

    def createInputForNN(self, games):
        x = []
        y = []
        for game in games:
            board = game.board()
            for move in game.mainline_moves():
                x.append(self.boardToMatrix(board))
                y.append(move.uci())
                board.push(move)
        return x, y

