import os
import tensorflow as tf
import numpy as np
import time
from chess import pgn, Board
from tqdm import tqdm
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
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


def data_generator(X, y, batch_size):
    dataset_size = len(X)
    indices = np.arange(dataset_size)
    while True:
        np.random.shuffle(indices)
        for i in range(0, dataset_size, batch_size):
            batch_indices = indices[i:i + batch_size]
            X_batch = tf.convert_to_tensor(X[batch_indices], dtype=tf.float32)
            y_batch = tf.convert_to_tensor(y[batch_indices], dtype=tf.float32)
            yield X_batch, y_batch

def train_existing_model(model_path, X, y, batch_size=32, epochs=50):
    print(f"\nLoading existing model from {model_path}")
    model = load_model(model_path)

    # Calculate steps per epoch
    steps_per_epoch = len(X) // batch_size

    # Create data generator
    train_generator = data_generator(X, y, batch_size)

    # Use legacy optimizer for M1/M2
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(),
        loss=model.loss,
        metrics=model.metrics
    )

    print("\nContinuing training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=1
    )

    print("\nSaving updated model...")
    model.save(model_path)
    return model, history
  def encode_moves(moves):
      unique_moves = sorted(set(moves))
      num_classes = len(unique_moves)
      move_to_int = {move: i for i, move in enumerate(unique_moves)}

      def encode_chunk(move_chunk):
          return [move_to_int[move] for move in move_chunk]

      chunk_size = 10000
      chunks = [moves[i:i + chunk_size] for i in range(0, len(moves), chunk_size)]

      encoded_moves = []
      with ThreadPoolExecutor(max_workers=8) as executor:
          results = list(tqdm(
              executor.map(encode_chunk, chunks),
              total=len(chunks),
              desc="Encoding moves",
              colour="blue"
          ))

      for chunk in results:
          encoded_moves.extend(chunk)

      return np.array(encoded_moves), move_to_int, num_classes

  def train_existing_model(model_path, X, y, num_classes, batch_size=32, epochs=50):
      print(f"\nLoading existing model from {model_path}")
      model = load_model(model_path)

      steps_per_epoch = len(X) // batch_size
      train_generator = data_generator(X, y, batch_size, num_classes)

      model.compile(
          optimizer=tf.keras.optimizers.legacy.Adam(),
          loss=model.loss,
          metrics=model.metrics
      )

      print("\nContinuing training...")
      history = model.fit(
          train_generator,
          steps_per_epoch=steps_per_epoch,
          epochs=epochs,
          verbose=1
      )

      print("\nSaving updated model...")
      model.save(model_path)
      return model, history

  def data_generator(X, y, batch_size, num_classes):
      dataset_size = len(X)
      indices = np.arange(dataset_size)
      while True:
          np.random.shuffle(indices)
          for i in range(0, dataset_size, batch_size):
              batch_indices = indices[i:i + batch_size]
              X_batch = tf.convert_to_tensor(X[batch_indices], dtype=tf.float32)
              y_batch = tf.convert_to_tensor(to_categorical(y[batch_indices], num_classes=num_classes), dtype=tf.float32)
              yield X_batch, y_batch

  def main():
      start_time = time.time()

      # Configuration
      MODEL_PATH = '../assets/chessModel.keras'
      FILE_PATH = '../assets/ChessData'
      LIMIT_OF_FILES = 2
      GAMES_LIMIT = 50000

      # Load games with proper initialization
      print("\nLoading games...")
      games = load_games_threaded(FILE_PATH, LIMIT_OF_FILES)

      if not games:  # Check if games were loaded
          print("No games were loaded. Check the file path and try again.")
          return

      print(f"\nTotal games loaded: {len(games)}")
      LIMITED_GAMES = games[:GAMES_LIMIT]

      # Prepare training data
      print("\nPreparing training data...")
      X, y = create_input_for_nn(LIMITED_GAMES)
      y, move_to_int, num_classes = encode_moves(y)
      X = np.array(X)

      print(f"\nTraining data shapes:")
      print(f"X shape: {X.shape}")
      print(f"y shape: {y.shape}")

      # Train model
      model, history = train_existing_model(MODEL_PATH, X, y, num_classes)

      # Create move mapping for predictions
      int_to_move = dict(zip(move_to_int.values(), move_to_int.keys()))

      # Print training time
    end_time = time.time()
    print(f"\nTotal training time: {(end_time - start_time) / 60:.2f} minutes")

    return model, history, int_to_move


if __name__ == "__main__":
    main()
