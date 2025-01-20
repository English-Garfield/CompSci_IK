import os
import tensorflow as tf
import numpy as np
import random as rdm
import time
from chess import pgn, Board
from tqdm import tqdm
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

os.environ['TF_METAL'] = '1'


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
    X, y = [], []
    chunk_size = 1000

    for i in range(0, len(games), chunk_size):
        chunk = games[i:i + chunk_size]
        for game in chunk:
            board = game.board()
            for move in game.mainline_moves():
                X.append(board_to_matrix(board))
                y.append(move.uci())
                board.push(move)

    return np.array(X, dtype=np.float32), np.array(y)


def encode_moves(moves):
    move_to_int = {move: idx for idx, move in enumerate(set(moves))}
    return [move_to_int[move] for move in moves], move_to_int


def data_generator(X, y, batch_size):
    dataset_size = len(X)
    while True:
        for i in range(0, dataset_size, batch_size):
            end = min(i + batch_size, dataset_size)
            yield X[i:end], y[i:end]


def train_existing_model(model_path, X, y, batch_size=32, epochs=25):
    print(f"\nLoading existing model from {model_path}")
    model = load_model(model_path)

    # Convert to TensorFlow tensors
    X = np.array(X, dtype=np.float32)
    X = tf.convert_to_tensor(X, dtype=tf.float32)

    y = np.array(y, dtype=np.int32)
    y = tf.convert_to_tensor(y, dtype=tf.int32)

    # Create data generator
    train_gen = data_generator(X, y, batch_size)

    # Use legacy optimizer for M1/M2
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(),
        loss=model.loss,
        metrics=model.metrics
    )

    print("\nContinuing training...")
    history = model.fit(
        train_gen,
        validation_split=0.1,
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
    LIMIT_OF_FILES = 3
    GAMES_LIMIT = 50000

    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        tf.config.set_visible_devices(physical_devices[0], 'GPU')

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
