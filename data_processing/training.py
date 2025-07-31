import os
import sys
import tensorflow as tf
import numpy as np
import time
import json
import gc
import argparse
import threading

from chess import pgn, Board
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

# tensorflow modules
from keras.src.layers import GlobalAveragePooling1D
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization,
    LeakyReLU, Add, Input, Multiply, Lambda, Attention, MultiHeadAttention,
    LayerNormalization, Concatenate, SeparableConv2D, DepthwiseConv2D,
    GlobalMaxPooling2D, Reshape, Permute, Activation
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

# Enable Metal GPU acceleration for Mac
os.environ['TF_METAL'] = '1'


def load_pgn(file_path):
    with open(file_path, 'r') as pgn_file:
        while True:
            game = pgn.read_game(pgn_file)
            if game is None:
                break
            yield game


def board_to_matrix_enhanced(board: Board):
    matrix = np.zeros((8, 8, 18))  # Increased channels for more representation
    piece_map = board.piece_map()

    # Basic piece positions (12 channels)
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[row, col, piece_type + piece_color] = 1

    # New features! (6 channels)
    # Channel 12: Attacked squares by white
    # Channel 13: Attacked squares by black
    # Channel 14: En passant target
    # Channel 15: Castling rights (encoded as a boolean map)
    # Channel 16: Turn indicator (1 for white, 0 for black)
    # Channel 17: Check indicator

    # Attacked squares
    white_attacks = np.zeros((8, 8))
    black_attacks = np.zeros((8, 8))

    for square in range(64):
        if board.is_attacked_by(True, square):  # White attacks
            row, col = divmod(square, 8)
            white_attacks[row, col] = 1
        if board.is_attacked_by(False, square):  # Black attacks
            row, col = divmod(square, 8)
            black_attacks[row, col] = 1

    matrix[:, :, 12] = white_attacks
    matrix[:, :, 13] = black_attacks

    # En passant
    if board.ep_square:
        row, col = divmod(board.ep_square, 8)
        matrix[row, col, 14] = 1

    # Castling rights (simplified)
    if board.has_castling_rights(True):  # White castling
        matrix[7, 4, 15] = 1  # King position
    if board.has_castling_rights(False):  # Black castling
        matrix[0, 4, 15] = 1  # King position

    # Turn indicator
    matrix[:, :, 16] = 1 if board.turn else 0

    # King check indicator
    if board.is_check():
        matrix[:, :, 17] = 1

    return matrix


def board_to_matrix(board: Board):
    return board_to_matrix_enhanced(board)[:, :, :12]  # Use only basic features for backward compatibility


def squeeze_excite_block(input_tensor, ratio=16):
    # Channel attention mechanism
    channels = input_tensor.shape[-1]

    # Squeeze
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, channels))(se)

    # Excitation
    se = Dense(channels // ratio, activation='relu', kernel_regularizer=l2(0.0001))(se)
    se = Dense(channels, activation='sigmoid', kernel_regularizer=l2(0.0001))(se)

    # Scale
    return Multiply()([input_tensor, se])


def spatial_attention_block(input_tensor):
    # Spacial mechanism
    # Average and max pooling across channels
    avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(input_tensor)
    max_pool = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(input_tensor)

    # Concatenate and apply convolution
    concat = Concatenate()([avg_pool, max_pool])
    attention = Conv2D(1, (7, 7), padding='same', activation='sigmoid')(concat)

    return Multiply()([input_tensor, attention])


def residual_block(x, filters, kernel_size=(3, 3), dropout_rate=0.1):
    # Residual Block
    shortcut = x

    # 1 convolution
    x = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(dropout_rate)(x)

    # 2 convolution
    x = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)

    # S&E block
    x = squeeze_excite_block(x)

    # Adjust shortcut (if needed)
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), padding='same', kernel_regularizer=l2(0.0001))(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # residual connection
    x = Add()([x, shortcut])
    x = LeakyReLU(alpha=0.1)(x)

    return x


def create_advanced_chess_model(input_shape, num_classes, model_type='resnet_attention', learning_rate=0.001):
    # enhanced model creation
    if model_type == 'resnet_attention':
        return create_resnet_attention_model(input_shape, num_classes, learning_rate)
    elif model_type == 'transformer':
        return create_transformer_model(input_shape, num_classes, learning_rate)
    elif model_type == 'efficientnet_style':
        return create_efficientnet_style_model(input_shape, num_classes, learning_rate)
    elif model_type == 'hybrid':
        return create_hybrid_model(input_shape, num_classes, learning_rate)
    else:
        return create_resnet_attention_model(input_shape, num_classes, learning_rate)


def create_resnet_attention_model(input_shape, num_classes, learning_rate=0.001):
    # ResNet with an attention mechanism
    print(f"\nCreating ResNet-Attention chess model with learning rate: {learning_rate}")

    inputs = Input(shape=input_shape)

    # Initial convolution
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.0001))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Residual blocks with increasing complexity
    x = residual_block(x, 64, dropout_rate=0.1)
    x = residual_block(x, 64, dropout_rate=0.1)

    x = residual_block(x, 128, dropout_rate=0.15)
    x = residual_block(x, 128, dropout_rate=0.15)

    x = residual_block(x, 256, dropout_rate=0.2)
    x = residual_block(x, 256, dropout_rate=0.2)

    # Spatial attention
    x = spatial_attention_block(x)

    # Global pooling
    gap = GlobalAveragePooling2D()(x)
    gmp = GlobalMaxPooling2D()(x)
    global_features = Concatenate()([gap, gmp])

    # Dense layers
    x = Dense(512, kernel_regularizer=l2(0.0001))(global_features)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.3)(x)

    x = Dense(256, kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.4)(x)

    # Output layer
    outputs = Dense(num_classes, activation='softmax', name='move_prediction')(x)

    model = Model(inputs, outputs)

    # AdamW optimizer
    optimizer = AdamW(learning_rate=learning_rate, weight_decay=0.0001)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )

    return model


def create_transformer_model(input_shape, num_classes, learning_rate=0.001):
    # Transformer Architecture
    print(f"\nCreating Transformer chess model with learning rate: {learning_rate}")

    inputs = Input(shape=input_shape)

    # embedding
    x = Conv2D(256, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    # flatten spatial dimensions
    batch_size = tf.shape(x)[0]
    x = Reshape((64, 256))(x)  # 8x8 -> 64 positions, 256 features each

    # Positional encoding
    position_encoding = tf.range(64, dtype=tf.float32)[tf.newaxis, :, tf.newaxis]
    position_encoding = tf.tile(position_encoding, [1, 1, 256])
    x = x + position_encoding * 0.01

    # Multi-head attention layers
    for i in range(4):
        attention_output = MultiHeadAttention(
            num_heads=8, key_dim=32, dropout=0.1
        )(x, x)

        # Add & Norm
        x = Add()([x, attention_output])
        x = LayerNormalization()(x)

        # Feed forward
        ff = Dense(512, activation='relu')(x)
        ff = Dropout(0.1)(ff)
        ff = Dense(256)(ff)

        x = Add()([x, ff])
        x = LayerNormalization()(x)

    x = GlobalAveragePooling1D()(x)

    # Final dense layers
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)

    outputs = Dense(num_classes, activation='softmax', name='move_prediction')(x)

    model = Model(inputs, outputs)

    optimizer = AdamW(learning_rate=learning_rate, weight_decay=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )

    return model


def create_efficientnet_style_model(input_shape, num_classes, learning_rate=0.001):
    # EfficientNet-style architecture
    print(f"\nCreating EfficientNet-style chess model with learning rate: {learning_rate}")

    inputs = Input(shape=input_shape)

    # Stem
    x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(0.0001))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    # MBConv blocks
    for i, (filters, repeats) in enumerate([(64, 2), (128, 3), (256, 4)]):
        for j in range(repeats):
            # Expansion
            expanded = Conv2D(filters * 4, (1, 1), padding='same')(x)
            expanded = BatchNormalization()(expanded)
            expanded = LeakyReLU(alpha=0.1)(expanded)

            # Depthwise
            dw = DepthwiseConv2D((3, 3), padding='same')(expanded)
            dw = BatchNormalization()(dw)
            dw = LeakyReLU(alpha=0.1)(dw)

            # Squeeze and Excitation
            se = squeeze_excite_block(dw, ratio=16)

            # Projection
            projected = Conv2D(filters, (1, 1), padding='same')(se)
            projected = BatchNormalization()(projected)

            # Skip the connection if it has the same dimensions
            if x.shape[-1] == filters:
                x = Add()([x, projected])
            else:
                x = projected

    # Head
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)

    outputs = Dense(num_classes, activation='softmax', name='move_prediction')(x)

    model = Model(inputs, outputs)

    optimizer = AdamW(learning_rate=learning_rate, weight_decay=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )

    return model


def create_hybrid_model(input_shape, num_classes, learning_rate=0.001):
    # Hybrid CNN-Transformer architecture
    print(f"\nCreating Hybrid chess model with learning rate: {learning_rate}")

    inputs = Input(shape=input_shape)

    # CNN branch
    cnn_branch = Conv2D(64, (3, 3), padding='same')(inputs)
    cnn_branch = BatchNormalization()(cnn_branch)
    cnn_branch = LeakyReLU(alpha=0.1)(cnn_branch)

    for filters in [64, 128, 256]:
        cnn_branch = residual_block(cnn_branch, filters)

    cnn_features = GlobalAveragePooling2D()(cnn_branch)

    # Position-aware branch
    pos_branch = Reshape((64, input_shape[-1]))(inputs)

    # Add positional embeddings
    pos_embedding = Dense(128, activation='relu')(pos_branch)
    pos_embedding = LayerNormalization()(pos_embedding)

    # Self-attention on positions
    attention_output = MultiHeadAttention(num_heads=4, key_dim=32)(pos_embedding, pos_embedding)
    pos_features = GlobalAveragePooling1D()(attention_output)

    # Combine branches
    combined = Concatenate()([cnn_features, pos_features])

    # Final layers
    x = Dense(512, activation='relu')(combined)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)

    outputs = Dense(num_classes, activation='softmax', name='move_prediction')(x)

    model = Model(inputs, outputs)

    optimizer = AdamW(learning_rate=learning_rate, weight_decay=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )

    return model


def create_new_model(input_shape, num_classes, learning_rate=0.003, batch_size=64, epochs=50,
                     model_type='resnet_attention'):
    # Create a new chess model based on user input
    return create_advanced_chess_model(input_shape, num_classes, model_type, learning_rate)


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


def process_game_chunk(game_chunk, use_enhanced_features=False):
    X, y = [], []
    for game in game_chunk:
        board = game.board()
        for move in game.mainline_moves():
            if use_enhanced_features:
                X.append(board_to_matrix_enhanced(board))
            else:
                X.append(board_to_matrix(board))
            y.append(move.uci())
            board.push(move)
    return X, y


def create_input_for_nn(games, chunk_size=500, use_enhanced_features=False):
    # Create input data for the NN from data
    print(f"\nProcessing games in chunks of {chunk_size} to optimize memory usage")
    print(f"Using enhanced features: {use_enhanced_features}")
    X, y = [], []

    num_threads = min(4, os.cpu_count())

    for i in range(0, len(games), chunk_size):
        print(f"\nProcessing chunk {i // chunk_size + 1}/{(len(games) + chunk_size - 1) // chunk_size}")
        chunk = games[i:i + chunk_size]

        sub_chunk_size = max(10, chunk_size // num_threads)
        sub_chunks = [chunk[j:j + sub_chunk_size] for j in range(0, len(chunk), sub_chunk_size)]

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(tqdm(
                executor.map(lambda sc: process_game_chunk(sc, use_enhanced_features), sub_chunks),
                total=len(sub_chunks),
                desc="Processing games",
                colour="green"
            ))

        for chunk_X, chunk_y in results:
            X.extend(chunk_X)
            y.extend(chunk_y)

        gc.collect()

    X_array = np.array(X, dtype=np.float32)
    y_array = np.array(y)

    X.clear()
    y.clear()
    gc.collect()

    return X_array, y_array


def data_generator(X, y, batch_size, num_classes):
    # Generator for training data
    num_samples = len(X)
    indices = np.arange(num_samples)

    while True:
        np.random.shuffle(indices)

        for offset in range(0, num_samples, batch_size):
            batch_indices = indices[offset:offset + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            y_one_hot = to_categorical(y_batch, num_classes=num_classes)

            yield X_batch, y_one_hot

            if offset % (batch_size * 10) == 0:
                gc.collect()


def create_learning_rate_scheduler(initial_lr, schedule_type='exponential'):
    # enhanced lr_scheduler
    if schedule_type == 'exponential':
        def exponential_decay(epoch, lr):
            if epoch < 10:
                return lr
            else:
                return lr * tf.math.exp(-0.1)

        return LearningRateScheduler(exponential_decay)

    elif schedule_type == 'step':
        def step_decay(epoch, lr):
            drop_rate = 0.5
            epochs_drop = 15.0
            return initial_lr * tf.math.pow(drop_rate, tf.math.floor(epoch / epochs_drop))

        return LearningRateScheduler(step_decay)

    elif schedule_type == 'cosine':
        def cosine_decay(epoch, lr):
            alpha = 0.0001
            cosine_decay = 0.5 * (1 + tf.math.cos(tf.math.pi * epoch / 50))
            decayed = (1 - alpha) * cosine_decay + alpha
            return initial_lr * decayed

        return LearningRateScheduler(cosine_decay)

    elif schedule_type == 'warmup_cosine':
        def warmup_cosine_decay(epoch, lr):
            warmup_epochs = 10
            total_epochs = 50

            if epoch < warmup_epochs:
                return initial_lr * (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return initial_lr * 0.5 * (1 + tf.math.cos(tf.math.pi * progress))

        return LearningRateScheduler(warmup_cosine_decay)


def get_user_input():
    # Enhanced user input code
    print("\n=== Enhanced Chess Intelligence Model Training Configuration ===")

    # Model architecture choice
    print("\n--- Model Architecture ---")
    print("Choose the model architecture:")
    print("1. ResNet with Attention (recommended)")
    print("2. Transformer-based")
    print("3. EfficientNet-style")
    print("4. Hybrid CNN-Transformer")

    while True:
        arch_choice = input("Enter your choice (1-4) [default: 1]: ").strip()
        if arch_choice == "" or arch_choice == "1":
            model_type = "resnet_attention"
            break
        elif arch_choice == "2":
            model_type = "transformer"
            break
        elif arch_choice == "3":
            model_type = "efficientnet_style"
            break
        elif arch_choice == "4":
            model_type = "hybrid"
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

    # Enhanced features option
    print("\n--- Board Representation ---")
    use_enhanced = input("Use enhanced board features (attacks, castling, etc.)? (y/n) [default: n]: ").strip().lower()
    use_enhanced_features = use_enhanced in ['y', 'yes', '1', 'true']

    # Rest of the existing input collection...
    # (keeping the same structure as original get_user_input function)

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

    # Learning rate configuration
    print("\n--- Learning Rate Configuration ---")
    while True:
        lr_input = input("Initial learning rate [default: 0.001]: ").strip()
        if lr_input == "":
            learning_rate = 0.001
            break
        try:
            learning_rate = float(lr_input)
            if learning_rate > 0:
                break
            else:
                print("Learning rate must be a positive number.")
        except ValueError:
            print("Please enter a valid number.")

    print("\nLearning rate schedule options:")
    print("1. None (constant learning rate)")
    print("2. ReduceLROnPlateau (automatic reduction when loss plateaus)")
    print("3. Exponential decay")
    print("4. Step decay")
    print("5. Cosine decay")
    print("6. Warmup + Cosine decay")

    while True:
        schedule_choice = input("Choose learning rate schedule (1-6) [default: 2]: ").strip()
        if schedule_choice == "" or schedule_choice == "2":
            lr_schedule_type = "plateau"
            break
        elif schedule_choice == "1":
            lr_schedule_type = "none"
            break
        elif schedule_choice == "3":
            lr_schedule_type = "exponential"
            break
        elif schedule_choice == "4":
            lr_schedule_type = "step"
            break
        elif schedule_choice == "5":
            lr_schedule_type = "cosine"
            break
        elif schedule_choice == "6":
            lr_schedule_type = "warmup_cosine"
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 6.")

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
    args.model_type = model_type
    args.use_enhanced_features = use_enhanced_features
    args.learning_rate = learning_rate
    args.lr_schedule = lr_schedule_type
    args.batch_size = batch_size
    args.file_limit = file_limit
    args.games_limit = games_limit
    args.chunk_size = chunk_size
    args.epochs = epochs
    args.model_path = model_path
    args.move_map_path = move_map_path
    args.data_path = data_path

    return args


def load_games_threaded(file_path, limit_of_files):
    files = [f"{file_path}/{file}" for file in os.listdir(file_path) if file.endswith('.pgn')]
    files = files[:limit_of_files]
    print(f"\nLoading {len(files)} PGN files from {file_path}")

    num_threads = min(2, len(files))
    chunks = np.array_split(files, num_threads)

    queue = Queue()
    threads = []

    def parallel_load_games(file_paths, queue):
        for file_path in file_paths:
            games = list(load_pgn(file_path))
            queue.put(games)

    for chunk in chunks:
        thread = threading.Thread(target=parallel_load_games, args=(chunk, queue))
        threads.append(thread)
        thread.start()

    # Collect results
    all_games = []
    for _ in threads:
        games = queue.get()
        all_games.extend(games)
        gc.collect()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print(f"Loaded {len(all_games)} games total")
    return all_games


def train_existing_model(model_path, X, y, num_classes, learning_rate=0.003, batch_size=64, epochs=50,
                         lr_schedule=None, model_type='resnet_attention'):

    # Load an existing model and continue training
    print(f"\nLoading existing model from {model_path}")
    model = load_model(model_path)

    model_output_classes = model.output_shape[-1]

    if model_output_classes != num_classes:
        print(f"Model output shape mismatch: Model outputs {model_output_classes}, but num_classes is {num_classes}.")
        print("Rebuilding the output layer to match the new number of classes...")

        intermediate_layer = model.layers[-2].output

        x = BatchNormalization()(intermediate_layer)
        x = Dropout(0.2)(x)
        new_output = Dense(num_classes, activation="softmax", name="move_prediction")(x)

        model = Model(inputs=model.input, outputs=new_output)

    # Update optimizer with a new learning rate
    optimizer = AdamW(learning_rate=learning_rate, weight_decay=0.0001)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", "top_k_categorical_accuracy"]
    )

    print(f"Updated model with learning rate: {learning_rate}")

    steps_per_epoch = len(X) // batch_size
    train_generator = data_generator(X, y, batch_size, num_classes)

    # Setup callbacks
    callbacks = []

    # Add a learning rate scheduler if specified
    if lr_schedule:
        callbacks.append(lr_schedule)

    # Add ReduceLROnPlateau as a backup
    reduce_lr = ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    callbacks.append(reduce_lr)

    # Add early stopping to prevent overfitting
    early_stop = EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stop)

    # Add model checkpointing
    checkpoint = ModelCheckpoint(
        model_path.replace('.keras', '_best.keras'),
        monitor='loss',
        save_best_only=True,
        verbose=1
    )
    callbacks.append(checkpoint)

    print("\nContinuing training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print("\nSaving updated model...")
    model.save(model_path)
    return model, history


def train_model(model, X, y, num_classes, batch_size=64, epochs=50, lr_schedule=None, model_path=None):
    steps_per_epoch = len(X) // batch_size
    train_generator = data_generator(X, y, batch_size, num_classes)

    # callbacks
    callbacks = []

    # learning rate scheduler if specified
    if lr_schedule:
        callbacks.append(lr_schedule)

    # ReduceLROnPlateau as a backup
    reduce_lr = ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    callbacks.append(reduce_lr)

    # early stopping to prevent overfitting
    early_stop = EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stop)

    #  model checkpointing if a path is provided
    if model_path:
        checkpoint = ModelCheckpoint(
            model_path.replace('.keras', '_best.keras'),
            monitor='loss',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)

    print("\nTraining model...")
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    return model, history


def parse_arguments():
    # enhanced args parser
    parser = argparse.ArgumentParser(description='Enhanced Chess Intelligence Model Training Script')

    # Model architecture choice
    parser.add_argument('--model-type', type=str,
                        choices=['resnet_attention', 'transformer', 'efficientnet_style', 'hybrid'],
                        default='resnet_attention',
                        help='Model architecture type (default: resnet_attention)')

    # Enhanced features option
    parser.add_argument('--use-enhanced-features', action='store_true',
                        help='Use enhanced board representation with additional features')

    # Model creation/training choice
    parser.add_argument('--mode', type=str, choices=['auto', 'new', 'continue'], default='auto',
                        help='Training mode: auto (detect), new (create new model), or continue (train existing model)')

    # Learning rate parameters
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Initial learning rate (default: 0.001)')
    parser.add_argument('--lr-schedule', type=str,
                        choices=['none', 'plateau', 'exponential', 'step', 'cosine', 'warmup_cosine'],
                        default='plateau',
                        help='Learning rate schedule type (default: plateau)')

    # Memory optimization parameters
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training (default: 32)')
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
        # If no args were provided, use interactive input
        print("\nNo command-line arguments provided. Switching to interactive mode.")
        return get_user_input()


def main():
    start_time = time.time()

    # Parse CLI args
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
    LEARNING_RATE = args.learning_rate
    LR_SCHEDULE = args.lr_schedule
    MODEL_TYPE = getattr(args, 'model_type', 'resnet_attention')
    USE_ENHANCED_FEATURES = getattr(args, 'use_enhanced_features', False)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    print(f"\n=== Enhanced Chess Model Training ===")
    print(f"Model Architecture: {MODEL_TYPE}")
    print(f"Enhanced Features: {USE_ENHANCED_FEATURES}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Batch Size: {BATCH_SIZE}")

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
    X, y = create_input_for_nn(LIMITED_GAMES, chunk_size=CHUNK_SIZE, use_enhanced_features=USE_ENHANCED_FEATURES)

    # Free up memory
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

    # Create the lrs
    lr_scheduler = None
    if LR_SCHEDULE != 'none' and LR_SCHEDULE != 'plateau':
        lr_scheduler = create_learning_rate_scheduler(LEARNING_RATE, LR_SCHEDULE)

    # input shape
    if USE_ENHANCED_FEATURES:
        input_shape = (8, 8, 18)  # Enhanced rep
    else:
        input_shape = (8, 8, 12)  # Standard matrix rep

    # create a new model or train an existing one
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
            model, history = train_existing_model(
                MODEL_PATH, X, y, num_classes,
                learning_rate=LEARNING_RATE,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                lr_schedule=lr_scheduler,
                model_type=MODEL_TYPE
            )
        else:
            if create_new and model_exists:
                print(f"\nWarning: Overwriting existing model at {MODEL_PATH}")
            else:
                print(f"\nCreating a new {MODEL_TYPE} model")

            # Create a new model
            model = create_new_model(
                input_shape, num_classes,
                learning_rate=LEARNING_RATE,
                model_type=MODEL_TYPE
            )

            print("\nModel Architecture Summary:")
            model.summary()

            model, history = train_model(
                model, X, y, num_classes,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                lr_schedule=lr_scheduler,
                model_path=MODEL_PATH
            )

            print(f"\nSaving model to {MODEL_PATH}")
            model.save(MODEL_PATH)

        # Force garbage collection after training
        gc.collect()
    except Exception as e:
        print(f"Error during model training: {e}")

    # Create reverse mapping for prediction
    int_to_move = {v: k for k, v in move_to_int.items()}

    end_time = time.time()
    print(f"\nTotal training time: {(end_time - start_time) / 60:.2f} minutes")
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Move mapping saved to: {MOVE_MAP_PATH}")

    # training metrics
    if history and hasattr(history, 'history'):
        final_loss = history.history['loss'][-1]
        final_accuracy = history.history['accuracy'][-1]
        print(f"Final training loss: {final_loss:.4f}")
        print(f"Final training accuracy: {final_accuracy:.4f}")

    #  learning rate
    if hasattr(model.optimizer, 'learning_rate'):
        final_lr = model.optimizer.learning_rate.numpy()
        print(f"Final learning rate: {final_lr}")

    return model, history, int_to_move


if __name__ == "__main__":
    main()

"""
Chess Intelligence Model Training Script with Advanced Architecture

The script is used to create, train, and save a chess model for my project for comp sci / SoM. 
It can create a new model from scratch or continue training the existing model(s).
The script is optimized to work with limited memory aprox 16GB and M4 architecture.
      
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

Command-line examples:
    # ResNet with attention (recommended)
    python enhanced_training.py --model-type resnet_attention --use-enhanced-features --learning-rate 0.001

    # Transformer model with enhanced features
    python enhanced_training.py --model-type transformer --use-enhanced-features --learning-rate 0.0005

    # Hybrid model for best of both worlds
    python enhanced_training.py --model-type hybrid --use-enhanced-features --lr-schedule warmup_cosine

    # EfficientNet-style for resource efficiency
    python enhanced_training.py --model-type efficientnet_style --batch-size 64
    
Training History:
- training round 1 -> time 912 minutes -> 50/50 epochs -> loss: 3.1480 -> accuracy: 0.2976 (started @ 0.2086) -> lr: 0.0010
- training round 2 (New Model)-> time 2820 minutes -> 100/100 epochs -> loss: 3.1480 -> accuracy: 0.2524 (started @ 0.2000) -> lr: 0.0010
"""
