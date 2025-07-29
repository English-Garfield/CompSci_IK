import os
import sys
import tensorflow as tf
import numpy as np
import time
import json
import gc
import argparse
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

# Enable Metal GPU acceleration for Mac
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


def create_input_for_nn(games, chunk_size=500):
    """
    Process games into input data for neural network, with memory optimization.
    
    Args:
        games: List of chess games to process
        chunk_size: Size of chunks to process at once to limit memory usage
        
    Returns:
        X: Numpy array of board states
        y: Numpy array of moves
    """
    print(f"\nProcessing games in chunks of {chunk_size} to optimize memory usage")
    X, y = [], []
    
    # Reduce number of threads for better memory usage
    num_threads = min(4, os.cpu_count())
    
    # Process games in smaller chunks
    for i in range(0, len(games), chunk_size):
        print(f"\nProcessing chunk {i//chunk_size + 1}/{(len(games) + chunk_size - 1)//chunk_size}")
        chunk = games[i:i + chunk_size]
        
        # Further divide each chunk for parallel processing
        sub_chunk_size = max(10, chunk_size // num_threads)
        sub_chunks = [chunk[j:j + sub_chunk_size] for j in range(0, len(chunk), sub_chunk_size)]
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(tqdm(
                executor.map(process_game_chunk, sub_chunks),
                total=len(sub_chunks),
                desc="Processing games",
                colour="green"
            ))
        
        # Extend data with results from this chunk
        for chunk_X, chunk_y in results:
            X.extend(chunk_X)
            y.extend(chunk_y)
        
        # Force garbage collection after processing each chunk
        gc.collect()
    
    # Convert to numpy arrays
    X_array = np.array(X, dtype=np.float32)
    y_array = np.array(y)
    
    # Clear original lists to free memory
    X.clear()
    y.clear()
    
    # Force garbage collection
    gc.collect()
    
    return X_array, y_array


def parallel_load_games(file_paths, queue):
    for file_path in file_paths:
        games = list(load_pgn(file_path))
        queue.put(games)


def load_games_threaded(file_path, limit_of_files):
    files = [f"{file_path}/{file}" for file in os.listdir(file_path) if file.endswith('.pgn')]
    
    # Limit number of files to process
    files = files[:limit_of_files]
    print(f"\nLoading {len(files)} PGN files from {file_path}")
    
    # Use fewer threads to reduce memory usage
    num_threads = min(2, len(files))
    chunks = np.array_split(files, num_threads)
    
    queue = Queue()
    threads = []
    
    # Start threads to load games
    for chunk in chunks:
        thread = threading.Thread(target=parallel_load_games, args=(chunk, queue))
        threads.append(thread)
        thread.start()
    
    # Collect results
    all_games = []
    for _ in threads:
        games = queue.get()
        all_games.extend(games)
        # Force garbage collection after each batch
        gc.collect()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    print(f"Loaded {len(all_games)} games total")
    return all_games


def data_generator(X, y, batch_size, num_classes):
    num_samples = len(X)
    indices = np.arange(num_samples)
    
    while True:
        # Shuffle indices each epoch for better training
        np.random.shuffle(indices)
        
        for offset in range(0, num_samples, batch_size):
            # Get indices for this batch
            batch_indices = indices[offset:offset + batch_size]
            
            # Extract batch data using indices
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            # Convert y to one-hot encoding
            y_one_hot = to_categorical(y_batch, num_classes=num_classes)
            
            yield X_batch, y_one_hot
            
            # Suggest garbage collection after yielding
            if offset % (batch_size * 10) == 0:
                gc.collect()


def create_new_model(input_shape, num_classes, batch_size=64, epochs=50):
    print("\nCreating new chess model...")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, Flatten

    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model.summary()
    return model


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


def train_model(model, X, y, num_classes, batch_size=64, epochs=50):
    steps_per_epoch = len(X) // batch_size
    train_generator = data_generator(X, y, batch_size, num_classes)

    lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    print("\nTraining model...")
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=1,
        callbacks=[lr_scheduler, early_stopping]
    )
    
    return model, history


def get_user_input():
    """Collect user input for training parameters."""
    print("\n=== Chess Intelligence Model Training Configuration ===")
    
    # Model creation/training choice
    print("\n--- Model Mode ---")
    print("Choose the training mode:")
    print("1. Auto (automatically detect whether to create new model or continue training)")
    print("2. New (create a new model, overwriting any existing one)")
    print("3. Continue (continue training an existing model)")
    
    while True:
        mode_choice = input("Enter your choice (1-3) [default: 1]: ").strip()
        if mode_choice == "":
            mode = "auto"
            break
        elif mode_choice == "1":
            mode = "auto"
            break
        elif mode_choice == "2":
            mode = "new"
            break
        elif mode_choice == "3":
            mode = "continue"
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 3.")
    
    # Memory optimization parameters
    print("\n--- Memory Optimization Parameters ---")
    
    while True:
        batch_size_input = input("Batch size for training [default: 32]: ").strip()
        if batch_size_input == "":
            batch_size = 32
            break
        try:
            batch_size = int(batch_size_input)
            if batch_size > 0:
                break
            else:
                print("Batch size must be a positive integer.")
        except ValueError:
            print("Please enter a valid integer.")
    
    while True:
        file_limit_input = input("Maximum number of PGN files to load [default: 5]: ").strip()
        if file_limit_input == "":
            file_limit = 5
            break
        try:
            file_limit = int(file_limit_input)
            if file_limit > 0:
                break
            else:
                print("File limit must be a positive integer.")
        except ValueError:
            print("Please enter a valid integer.")
    
    while True:
        games_limit_input = input("Maximum number of games to process [default: 50000]: ").strip()
        if games_limit_input == "":
            games_limit = 50000
            break
        try:
            games_limit = int(games_limit_input)
            if games_limit > 0:
                break
            else:
                print("Games limit must be a positive integer.")
        except ValueError:
            print("Please enter a valid integer.")
    
    while True:
        chunk_size_input = input("Number of games to process in each chunk [default: 500]: ").strip()
        if chunk_size_input == "":
            chunk_size = 500
            break
        try:
            chunk_size = int(chunk_size_input)
            if chunk_size > 0:
                break
            else:
                print("Chunk size must be a positive integer.")
        except ValueError:
            print("Please enter a valid integer.")
    
    # Training parameters
    print("\n--- Training Parameters ---")
    
    while True:
        epochs_input = input("Number of training epochs [default: 50]: ").strip()
        if epochs_input == "":
            epochs = 50
            break
        try:
            epochs = int(epochs_input)
            if epochs > 0:
                break
            else:
                print("Epochs must be a positive integer.")
        except ValueError:
            print("Please enter a valid integer.")
    
    model_path = input("Path to save/load the model [default: ../assets/chessModel.keras]: ").strip()
    if model_path == "":
        model_path = "../assets/chessModel.keras"
    
    move_map_path = input("Path to save/load the move mapping [default: move_map.json]: ").strip()
    if move_map_path == "":
        move_map_path = "move_map.json"
    
    data_path = input("Path to the directory containing PGN files [default: ../assets/ChessData]: ").strip()
    if data_path == "":
        data_path = "../assets/ChessData"
    
    # Create a namespace object to mimic argparse's return value
    class Args:
        pass
    
    args = Args()
    args.mode = mode
    args.batch_size = batch_size
    args.file_limit = file_limit
    args.games_limit = games_limit
    args.chunk_size = chunk_size
    args.epochs = epochs
    args.model_path = model_path
    args.move_map_path = move_map_path
    args.data_path = data_path
    
    return args

def parse_arguments():
    """Parse command line arguments for the training script or use interactive input."""
    parser = argparse.ArgumentParser(description='Chess Intelligence Model Training Script')
    
    # Model creation/training choice
    parser.add_argument('--mode', type=str, choices=['auto', 'new', 'continue'], default='auto',
                        help='Training mode: auto (detect), new (create new model), or continue (train existing model)')
    
    # Memory optimization parameters
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training (default: 32, use smaller values for limited memory)')
    parser.add_argument('--file-limit', type=int, default=5,
                        help='Maximum number of PGN files to load (default: 5)')
    parser.add_argument('--games-limit', type=int, default=50000,
                        help='Maximum number of games to process (default: 50000)')
    parser.add_argument('--chunk-size', type=int, default=500,
                        help='Number of games to process in each chunk (default: 500)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--model-path', type=str, default='../assets/chessModel.keras',
                        help='Path to save/load the model (default: ../assets/chessModel.keras)')
    parser.add_argument('--move-map-path', type=str, default='move_map.json',
                        help='Path to save/load the move mapping (default: move_map.json)')
    parser.add_argument('--data-path', type=str, default='../assets/ChessData',
                        help='Path to the directory containing PGN files (default: ../assets/ChessData)')
    
    if len(sys.argv) > 1:
        return parser.parse_args()
    else:
        # If no arguments were provided, use interactive input
        print("\nNo command-line arguments provided. Switching to interactive mode.")
        return get_user_input()


def main():
    start_time = time.time()
    
    # Parse command line arguments
    args = parse_arguments()
    
    MODEL_PATH = args.model_path
    MOVE_MAP_PATH = args.move_map_path
    FILE_PATH = args.data_path
    LIMIT_OF_FILES = args.file_limit
    GAMES_LIMIT = args.games_limit
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    CHUNK_SIZE = args.chunk_size
    MODE = args.mode
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    print("\nLoading games...")
    games = load_games_threaded(FILE_PATH, LIMIT_OF_FILES)
    if not games:
        print("No games were loaded. Check the file path and try again.")
        return

    print(f"\nTotal games loaded: {len(games)}")
    LIMITED_GAMES = games[:GAMES_LIMIT]

    # Free up memory after limiting games
    games = None
    gc.collect()

    print("\nPreparing training data...")
    X, y = create_input_for_nn(LIMITED_GAMES, chunk_size=CHUNK_SIZE)
    
    # Free up memory after processing games
    LIMITED_GAMES = None
    gc.collect()
    
    if len(X) == 0 or len(y) == 0:
        print("\nNo training data available.")
        return

    y, move_to_int, num_classes = encode_moves(y, save_path=MOVE_MAP_PATH)

    print(f"\nTraining data shapes:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Number of unique moves (classes): {num_classes}")
    print(f"Estimated memory usage: {X.nbytes / (1024 * 1024):.2f} MB for X, {y.nbytes / (1024 * 1024):.2f} MB for y")

    model_exists = os.path.exists(MODEL_PATH)
    
    # Determine whether to create new model or train existing based on user choice
    create_new = False
    if MODE == 'new':
        create_new = True
        print("\nUser requested to create a new model")
    elif MODE == 'continue':
        if not model_exists:
            print(f"\nWarning: User requested to continue training but no model found at {MODEL_PATH}")
            print("Creating a new model instead")
            create_new = True
        else:
            create_new = False
            print("\nUser requested to continue training existing model")
    else:  # MODE == 'auto'
        create_new = not model_exists
        print(f"\nAuto mode: {'Creating new model' if create_new else 'Continuing training existing model'}")
    
    try:
        if not create_new and model_exists:
            print(f"\nLoading existing model from {MODEL_PATH}")
            model, history = train_existing_model(MODEL_PATH, X, y, num_classes, batch_size=BATCH_SIZE, epochs=EPOCHS)
        else:
            if create_new and model_exists:
                print(f"\nWarning: Overwriting existing model at {MODEL_PATH}")
            else:
                print(f"\nCreating a new model")
                
            # Create a new model
            input_shape = (8, 8, 12)  # Board representation shape
            model = create_new_model(input_shape, num_classes)
            
            # Train the new model
            model, history = train_model(model, X, y, num_classes, batch_size=BATCH_SIZE, epochs=EPOCHS)
            
            print(f"\nSaving model to {MODEL_PATH}")
            model.save(MODEL_PATH)
            
        # Force garbage collection after training
        gc.collect()
    except Exception as e:
        print(f"Error during model training: {e}")
        return

    # Create reverse mapping for prediction
    int_to_move = {v: k for k, v in move_to_int.items()}

    end_time = time.time()
    print(f"\nTotal training time: {(end_time - start_time) / 60:.2f} minutes")
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Move mapping saved to: {MOVE_MAP_PATH}")

    return model, history, int_to_move


if __name__ == "__main__":
    main()

"""
Chess Intelligence Model Training Script

The script is used to create, train, and save the chess model. 
It can either create a new model from scratch or continue training the existing model.
The script is optimized to work with limited memory
      
        Command-line Mode (For automation or advanced users):
          python training.py [options]
          Available options:
          --mode {auto,new,continue}  Choose whether to create a new model or train an existing one
                                      - auto: Automatically detect (default)
                                      - new: Create a new model (overwrites existing)
                                      - continue: Continue training an existing model
          
          Memory optimization options:
          --batch-size INT            Batch size for training (default: 32)
          --file-limit INT            Maximum number of PGN files to load (default: 5)
          --games-limit INT           Maximum number of games to process (default: 50000)
          --chunk-size INT            Number of games to process in each chunk (default: 500)
          
          Training parameters:
          --epochs INT                Number of training epochs (default: 50)
          --model-path PATH           Path to save/load the model (default: ../assets/chessModel.keras)
          --move-map-path PATH        Path to save/load the move mapping (default: move_map.json)
          --data-path PATH            Path to the directory containing PGN files (default: ../assets/ChessData)
       
    Examples:
       # Run in interactive mode
       python training.py
       
       # Create a new model with default settings
       python training.py --mode new
             
       # Continue training an existing model
       python training.py --mode continue
       
       # Use even more memory-efficient settings for very limited RAM
       python training.py --batch-size 16 --file-limit 2 --games-limit 20000 --chunk-size 200
       
  
Training History:
- training round 1 -> time 912 minutes -> 50/50 epochs -> loss: 3.1480 -> accuracy: 0.2976 (started @ 0.2086) -> lr: 0.0010
- training round 2 (New Model)-> time 2820 minutes -> 100/100 epochs -> loss: 3.1480 -> accuracy: 0.2524 (started @ 0.2000) -> lr: 0.0010
"""