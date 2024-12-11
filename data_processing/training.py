import os
import tensorflow as tf
import numpy as np
import random as rdm
import time
from chess import pgn, Board
from tqdm import tqdm
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical


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


def train_existing_model(model_path, X, y, batch_size=32, epochs=50):
    print(f"\nLoading existing model from {model_path}")
    model = load_model(model_path)

    print("\nContinuing training...")
    history = model.fit(
        X, y,
        validation_split=0.1,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )

    print("\nSaving updated model...")
    model.save(model_path)
    return model, history


def main():
    start_time = time.time()

    # Configuration
    MODEL_PATH = '../assets/chessModel.keras'
    FILE_PATH = '../assets/ChessData'
    LIMIT_OF_FILES = 5
    GAMES_LIMIT = 25000

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # Load games
    print("\nLoading games...")
    files = [file for file in os.listdir(FILE_PATH) if file.endswith('.pgn')]
    games = []

    for i, file in enumerate(tqdm(files)):
        games.extend(load_pgn(f"{FILE_PATH}/{file}"))
        if i >= LIMIT_OF_FILES - 1:
            break

    print(f"\nTotal games loaded: {len(games)}")
    LIMITED_GAMES = games[:GAMES_LIMIT]

    # Prepare training data
    print("\nPreparing training data...")
    X, y = create_input_for_nn(LIMITED_GAMES)
    y, move_to_int = encode_moves(y)
    y = to_categorical(y, num_classes=len(move_to_int))
    X = np.array(X)

    print(f"\nTraining data shapes:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # Train model
    model, history = train_existing_model(MODEL_PATH, X, y)

    # Create move mapping for predictions
    int_to_move = dict(zip(move_to_int.values(), move_to_int.keys()))

    # Print training time
    end_time = time.time()
    print(f"\nTotal training time: {(end_time - start_time) / 60:.2f} minutes")

    return model, history, int_to_move


if __name__ == "__main__":
    main()
