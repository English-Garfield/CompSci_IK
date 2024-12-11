import os
import tensorflow as tf
import numpy as np
import time

from chess import pgn
from tqdm import tqdm
from chess import Board
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.utils import to_categorical

# Start time
start_time = time.time()


def load_pgn(file_path):
    with open(file_path, 'r') as pgn_file:
        while True:
            game = pgn.read_game(pgn_file)
            if game is None:
                break
            yield game


def board_to_matrix(board: Board):
    matrix = np.zeros((8, 8, 12))
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[row, col, piece_type + piece_color] = 1
    return matrix


def create_input_for_nn(games):
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


def predict_next_move(board):
    board_matrix = board_to_matrix(board).reshape(1, 8, 8, 12)
    predictions = model.predict(board_matrix)[0]
    legal_moves = list(board.legal_moves)
    legal_moves_uci = [move.uci() for move in legal_moves]
    sorted_indices = np.argsort(predictions)[::-1]
    for move_index in sorted_indices:
        move = int_to_move[move_index]
        if move in legal_moves_uci:
            return move
    return None


# Main execution
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Loading model...")

# Record time for loading model
model_load_start = time.time()

files = [file for file in os.listdir('../assets/ChessData') if file.endswith('.pgn')]
print(files)
model_load_end = time.time()
print(f"Model loading time: {(model_load_end - model_load_start) / 60:.2f} minutes")
print("\nLoading games...")

LIMIT_OF_FILES = min(len(files), 10)
file_path = '../assets/ChessData'
games = []

# Record time for loading games
games_load_start = time.time()
for i, file in enumerate(tqdm(files)):
    games.extend(load_pgn(f"{file_path}/{file}"))
    if i >= LIMIT_OF_FILES - 1:
        break
games_load_end = time.time()
print(f"Games loading time: {(games_load_end - games_load_start) / 60:.2f} minutes")

print("\nGames loaded:", len(games))
print("Building and training neural network...")

# Limit the number of games processed
LIMITED_GAMES = games[:75000]  # Adjust the limit as needed

# Record time for creating input
input_creation_start = time.time()
X, y = create_input_for_nn(LIMITED_GAMES)
y, move_to_int = encode_moves(y)
y = to_categorical(y, num_classes=len(move_to_int))
X = np.array(X)
input_creation_end = time.time()
print(f"Input creation time: {(input_creation_end - input_creation_start) / 60:.2f} minutes")

print("\nX shape:", X.shape)
print("y shape:", y.shape)

# Record time for model building
model_build_start = time.time()
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(8, 8, 12)),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(len(move_to_int), activation='softmax')
])
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
model_build_end = time.time()
print(f"Model building time: {(model_build_end - model_build_start) / 60:.2f} minutes")

# Record time for model training
training_start = time.time()
model.fit(X, y, epochs=50,
          validation_split=0.1,
          batch_size=64)
training_end = time.time()
print(f"Model training time: {(training_end - training_start) / 60:.2f} minutes")

# Record time for model saving
model_save_start = time.time()
model.save('../assets/chessModelV2.keras')
model_save_end = time.time()
print(f"Model saving time: {(model_save_end - model_save_start) / 60:.2f} minutes")

int_to_move = dict(zip(move_to_int.values(), move_to_int.keys()))

# Test prediction
print("\nPredicting next move...")
board = Board()
print("Board before prediction:")
print(board)

# Record time for prediction
prediction_start = time.time()
next_move = predict_next_move(board)
prediction_end = time.time()
print(f"Prediction time: {(prediction_end - prediction_start) / 60:.2f} minutes")

board.push_uci(next_move)

print("\nPredicted move:", next_move)
print("Board after prediction:")
print(board)
print(str(pgn.Game.from_board(board)))

# End time
end_time = time.time()
total_time = end_time - start_time
print(f"\nTotal run time: {total_time / 60:.2f} minutes")