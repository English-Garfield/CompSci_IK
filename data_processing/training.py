import os
import tensorflow as tf
import numpy as np
import time
import json
from chess import pgn, Board
from tqdm import tqdm
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading

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


def encode_moves(moves, save_path="move_map.json"):
    unique_moves = sorted(set(moves))
    num_classes = len(unique_moves)
    move_to_int = {move: i for i, move in enumerate(unique_moves)}

    with open(save_path, "w") as f:
        json.dump({"move_to_int": move_to_int, "num_classes": num_classes}, f)

    return np.array([move_to_int[move] for move in moves]), move_to_int, num_classes


def load_move_mapping(save_path="move_map.json"):
    with open(save_path, "r") as f:
        data = json.load(f)
    return data["move_to_int"], data["num_classes"]


def process_game_chunk(game_chunk):
    X, y = [], []
    for game in game_chunk:
        board = game.board()
        for move in game.mainline_moves():
            X.append(board_to_matrix(board))
            y.append(move.uci())
            board.push(move)
    return X, y


def create_input_for_nn(games):
    X, y = [], []
    chunk_size = 1000
    num_threads = min(8, os.cpu_count())
    chunks = [games[i:i + chunk_size] for i in range(0, len(games), chunk_size)]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(tqdm(
            executor.map(process_game_chunk, chunks),
            total=len(chunks),
            desc="Processing games",
            colour="green"
        ))

    for chunk_X, chunk_y in results:
        X.extend(chunk_X)
        y.extend(chunk_y)

    return np.array(X, dtype=np.float32), np.array(y)


def parallel_load_games(file_paths, queue):
    for file_path in file_paths:
        games = list(load_pgn(file_path))
        queue.put(games)


def load_games_threaded(file_path, limit_of_files):
    files = [f"{file_path}/{file}" for file in os.listdir(file_path) if file.endswith('.pgn')][:limit_of_files]
    num_threads = min(4, len(files))
    chunks = np.array_split(files, num_threads)

    queue = Queue()
    threads = []

    for chunk in chunks:
        thread = threading.Thread(target=parallel_load_games, args=(chunk, queue))
        threads.append(thread)
        thread.start()

    all_games = []
    for _ in threads:
        games = queue.get()
        all_games.extend(games)

    for thread in threads:
        thread.join()

    return all_games


def data_generator(X, y, batch_size, num_classes):
    num_samples = len(X)
    while True:
        for offset in range(0, num_samples, batch_size):
            X_batch = X[offset:offset + batch_size]
            y_batch = y[offset:offset + batch_size]
            yield X_batch, to_categorical(y_batch, num_classes=num_classes)


def train_existing_model(model_path, X, y, num_classes, batch_size=64, epochs=50):
    print(f"\nLoading existing model from {model_path}")
    model = load_model(model_path)

    model_output_classes = model.output_shape[-1]

    if model_output_classes != num_classes:
        print(f"Model output shape mismatch: Model outputs {model_output_classes}, but num_classes is {num_classes}.")
        print("Rebuilding the output layer to match the new number of classes...")

        intermediate_layer = model.layers[-2].output

        x = BatchNormalization()(intermediate_layer)
        x = Dropout(0.2)(x)
        new_output = Dense(num_classes, activation="softmax", name="dense_output")(x)

        model = Model(inputs=model.input, outputs=new_output)

        optimizer = Adam(clipnorm=1.0)
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

    steps_per_epoch = len(X) // batch_size
    train_generator = data_generator(X, y, batch_size, num_classes)

    lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    print("\nContinuing training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=1,
        callbacks=[lr_scheduler, early_stopping]
    )

    print("\nSaving updated model...")
    model.save(model_path)
    return model, history


def main():
    start_time = time.time()

    MODEL_PATH = '../assets/chessModel.keras'
    FILE_PATH = '../assets/ChessData'
    LIMIT_OF_FILES = 2
    GAMES_LIMIT = 50000

    print("\nLoading games...")
    games = load_games_threaded(FILE_PATH, LIMIT_OF_FILES)
    if not games:
        print("No games were loaded. Check the file path and try again.")
        return

    print(f"\nTotal games loaded: {len(games)}")
    LIMITED_GAMES = games[:GAMES_LIMIT]

    print("\nPreparing training data...")
    X, y = create_input_for_nn(LIMITED_GAMES)
    if len(X) == 0 or len(y) == 0:
        print("\nNo training data available.")
        return

    y, move_to_int, num_classes = encode_moves(y)
    X = np.array(X)

    move_to_int, num_classes = load_move_mapping()

    print(f"\nTraining data shapes:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    try:
        model, history = train_existing_model(MODEL_PATH, X, y, num_classes)
    except ValueError as e:
        print(e)
        return

    int_to_move = {v: k for k, v in move_to_int.items()}

    end_time = time.time()
    print(f"\nTotal training time: {(end_time - start_time) / 60:.2f} minutes")

    return model, history, int_to_move


if __name__ == "__main__":
    user = input("Would you like to start training? (y/n): ")
    multi = input("Would you like to train multiple times? (y/n): ")

    if user.lower() == 'y' and multi.lower() == 'y':
        multiNum = int(input("How many times would you like to train? (int): "))
        print(f"Training {multiNum} times")
        for i in range(multiNum):
            main()
            print("training complete")
    elif user.lower() == 'y' and multi.lower() == 'n':
        main()
        print("training complete")
    else:
        print("training aborted \nFinished")
