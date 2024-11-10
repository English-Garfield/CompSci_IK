"""
This file is responsible for the AI of the chess game.
"""

import os
import numpy as np
import time

from chess import Board
from chess import pgn
from tqdm import tqdm
from const import *

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam


def board_to_matrix(board: Board):
    matrix = np.zeros((8, 8, 12))
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        ROWS, COLS = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[ROWS, COLS, piece_type + piece_color] = 1
    return matrix


def createInputForNN(games):
    X = []
    y = []
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            X.append(board_to_matrix(board))
            y.append(move.uci())
            board.push(move)
    return X, y


def encode_moves(moves):
    move_to_int = {move: idx for idx, move in enumerate(set(moves))}
    return [move_to_int[move] for move in moves], move_to_int


def PredictNextMove(board):
    board_matrix = board_to_matrix(board).reshape(1, 8, 8, 12)
    prediction = model.predict(board_matrix)[0]
    legal_moves = list(board.legal_moves)
    legal_moves_uci = [move.uci() for move in legal_moves]
    sorted_indices = np.argsort(prediction)[::-1]
    for move_index in sorted_indices:
        move = int_to_move[move_index]
        if move in legal_moves_uci:
            return move
    return None


games = [open('ProcessedChessData.txt', 'r').read()]

"""Training"""
print("Start of training")
training_start_time = time.time()
X, y = createInputForNN(games)
y, move_to_int = encode_moves(y)
y = to_categorical(y, num_classes=len(move_to_int))
X = np.array(X)

model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(8, 8, 12)),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(len(move_to_int), activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X, y, epochs=50, validation_split=0.1, batch_size=64)

model.save('../assets/engines/ChessTF_50EPOCHS.keras')

end_time = time.time()
print("\n")
print(f"End of training. Time taken: {end_time - training_start_time} seconds")

"""Predictions"""
print("Start of prediction")
predictions_start_time = time.time()

model = load_model('../assets/engines/ChessTF_50EPOCHS.keras')
int_to_move = dict(zip(move_to_int.values(), move_to_int.keys()))

board = Board()

print("Board before prediction")
print(board)

next_move = PredictNextMove(board)
if next_move:
    board.push_uci(next_move)
    print("\nPredicted move:", next_move)
    print("Board after prediction:")
    print(board)
    print(str(pgn.Game.from_board(board)))
else:
    print("No valid move predicted.")

end_time = time.time()
print("\n")
print(f"End of prediction. Time taken: {end_time - predictions_start_time} seconds")
print(f"total run time: {end_time - Program_start_time} seconds")
