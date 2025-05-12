import numpy as np
import os
import random
from resource_path import resource_path

# Try different import strategies
try:
    import tensorflow as tf
    try:
        print(f"TensorFlow version: {tf.__version__}")
    except AttributeError:
        print("TensorFlow imported, but version information is not available")
    TF_AVAILABLE = True
    # In TensorFlow 2.x, Keras is available as tf.keras
    try:
        # Check if keras is available in tensorflow
        if hasattr(tf, 'keras'):
            import tensorflow.keras as keras
            try:
                print(f"Keras version (from TensorFlow): {keras.__version__}")
            except AttributeError:
                print("Keras (from TensorFlow) imported, but version information is not available")
            KERAS_AVAILABLE = True
        else:
            # Try standalone keras as fallback
            import keras
            try:
                print(f"Keras version (standalone): {keras.__version__}")
            except AttributeError:
                print("Keras (standalone) imported, but version information is not available")
            KERAS_AVAILABLE = True
    except ImportError as e:
        KERAS_AVAILABLE = False
        print(f"Warning: Keras not available, will use random moves. Error: {e}")
except ImportError as e:
    TF_AVAILABLE = False
    KERAS_AVAILABLE = False
    print(f"Warning: TensorFlow not available, will use random moves. Error: {e}")
    # Try standalone keras as a last resort
    try:
        import keras
        try:
            print(f"Keras version (standalone): {keras.__version__}")
        except AttributeError:
            print("Keras (standalone) imported, but version information is not available")
        KERAS_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Standalone Keras not available either, will use random moves. Error: {e}")


from board import Board
from move import Move
from piece import Piece


class AIPlayer:
    def __init__(self):
        self.board = Board()
        self.model = None

        # Try different methods to load the model
        print("Starting model loading process...")
        # First try the path with subdirectory
        model_path = resource_path('assets/chessModel.keras')
        print(f"Trying path: {model_path}, exists: {os.path.exists(model_path)}")
        # If not found, try the path without subdirectory
        if not os.path.exists(model_path):
            model_path = resource_path('chessModel.keras')
            print(f"Trying path: {model_path}, exists: {os.path.exists(model_path)}")
            if not os.path.exists(model_path):
                # Try direct path in assets directory
                model_path = resource_path('assets') + '/chessModel.keras'
                print(f"Trying path: {model_path}, exists: {os.path.exists(model_path)}")
                if not os.path.exists(model_path):
                    # Try one more path at the root of assets
                    model_path = os.path.join(os.path.dirname(os.path.abspath(".")), 'assets', 'chessModel.keras')
                    print(f"Trying path: {model_path}, exists: {os.path.exists(model_path)}")

        if os.path.exists(model_path):
            print(f"Found model at: {model_path}")
            if TF_AVAILABLE and hasattr(tf, 'keras'):
                print("TensorFlow and Keras are available, attempting to load model...")
                try:
                    # Try loading with compile=False to avoid compilation errors
                    print(f"Loading model from {model_path} with tensorflow.keras (compile=False)...")
                    self.model = tf.keras.models.load_model(model_path, compile=False)
                    print("Model loaded successfully with tensorflow.keras")
                    # Check if the model has the expected methods
                    if hasattr(self.model, 'predict'):
                        print("Model has predict method")
                    else:
                        print("WARNING: Model does not have predict method!")
                except Exception as e:
                    print(f"Failed to load model with tensorflow.keras: {e}")
                    # Try loading with h5py if available
                    try:
                        print("Attempting to load with h5py...")
                        import h5py
                        try:
                            print(f"h5py version: {h5py.__version__}")
                        except AttributeError:
                            print("h5py imported, but version information is not available")
                        # Try loading as h5 file
                        print(f"Loading model from {model_path} with tensorflow.keras and h5py...")
                        self.model = tf.keras.models.load_model(model_path, compile=False)
                        print("Model loaded successfully with tensorflow.keras and h5py")
                        # Check if the model has the expected methods
                        if hasattr(self.model, 'predict'):
                            print("Model has predict method")
                        else:
                            print("WARNING: Model does not have predict method!")
                    except Exception as h5_error:
                        print(f"Failed to load model with h5py: {h5_error}")
                        # Try loading as SavedModel format
                        try:
                            print("Attempting to load as SavedModel format...")
                            # Check if the path is a directory (SavedModel format)
                            if os.path.isdir(model_path):
                                print(f"Loading model from directory {model_path} as SavedModel...")
                                self.model = tf.saved_model.load(model_path)
                                print("Model loaded successfully as SavedModel")
                                # Check if the model has the expected methods
                                if hasattr(self.model, 'predict'):
                                    print("Model has predict method")
                                else:
                                    print("WARNING: Model does not have predict method!")
                            else:
                                # Try changing the extension to .h5
                                h5_path = model_path.replace('.keras', '.h5')
                                print(f"Trying alternative path with .h5 extension: {h5_path}, exists: {os.path.exists(h5_path)}")
                                if os.path.exists(h5_path):
                                    print(f"Loading model from {h5_path} with tensorflow.keras...")
                                    self.model = tf.keras.models.load_model(h5_path, compile=False)
                                    print(f"Model loaded successfully from alternative path: {h5_path}")
                                    # Check if the model has the expected methods
                                    if hasattr(self.model, 'predict'):
                                        print("Model has predict method")
                                    else:
                                        print("WARNING: Model does not have predict method!")
                        except Exception as sm_error:
                            print(f"Failed to load model as SavedModel: {sm_error}")

            if self.model is None and KERAS_AVAILABLE:
                print("Model not loaded with TensorFlow, trying standalone Keras...")
                try:
                    # Try loading with compile=False to avoid compilation errors
                    print(f"Loading model from {model_path} with standalone keras (compile=False)...")
                    self.model = keras.models.load_model(model_path, compile=False)
                    print("Model loaded successfully with keras")
                    # Check if the model has the expected methods
                    if hasattr(self.model, 'predict'):
                        print("Model has predict method")
                    else:
                        print("WARNING: Model does not have predict method!")
                except Exception as e:
                    print(f"Failed to load model with keras: {e}")
                    # Try loading with h5py if available
                    try:
                        print("Attempting to load with h5py and standalone keras...")
                        import h5py
                        try:
                            print(f"h5py version: {h5py.__version__}")
                        except AttributeError:
                            print("h5py imported, but version information is not available")
                        # Try loading as h5 file
                        print(f"Loading model from {model_path} with standalone keras and h5py...")
                        self.model = keras.models.load_model(model_path, compile=False)
                        print("Model loaded successfully with keras and h5py")
                        # Check if the model has the expected methods
                        if hasattr(self.model, 'predict'):
                            print("Model has predict method")
                        else:
                            print("WARNING: Model does not have predict method!")
                    except Exception as h5_error:
                        print(f"Failed to load model with h5py and standalone keras: {h5_error}")
                        # Try loading as SavedModel format
                        try:
                            print("Attempting to load as SavedModel format with standalone keras...")
                            # Check if the path is a directory (SavedModel format)
                            if os.path.isdir(model_path):
                                print(f"Loading model from directory {model_path} as SavedModel with standalone keras...")
                                # For standalone Keras, we can't use tf.saved_model.load
                                # Try to use keras.models.load_model with a different approach
                                self.model = keras.models.load_model(model_path, compile=False)
                                print("Model loaded successfully as SavedModel with keras")
                                # Check if the model has the expected methods
                                if hasattr(self.model, 'predict'):
                                    print("Model has predict method")
                                else:
                                    print("WARNING: Model does not have predict method!")
                            else:
                                # Try changing the extension to .h5
                                h5_path = model_path.replace('.keras', '.h5')
                                print(f"Trying alternative path with .h5 extension: {h5_path}, exists: {os.path.exists(h5_path)}")
                                if os.path.exists(h5_path):
                                    print(f"Loading model from {h5_path} with standalone keras...")
                                    self.model = keras.models.load_model(h5_path, compile=False)
                                    print(f"Model loaded successfully from alternative path: {h5_path}")
                                    # Check if the model has the expected methods
                                    if hasattr(self.model, 'predict'):
                                        print("Model has predict method")
                                    else:
                                        print("WARNING: Model does not have predict method!")
                        except Exception as sm_error:
                            print(f"Failed to load model as SavedModel with keras: {sm_error}")

            if self.model is None:
                print("Warning: Could not load AI model, will use random moves")
        else:
            print(f"Model file not found at any of the expected locations. Tried: {model_path}")
            print("Warning: Could not load AI model, will use random moves")

    def get_move(self, board):
        # Get valid moves for the AI's pieces
        valid_moves = self.get_valid_moves(board)

        # Debugging: Output the number of valid moves
        print(f"Number of valid moves: {len(valid_moves)}")

        if not valid_moves:
            print("No valid moves available! Assuming checkmate.")
            return None

        # If model is not available, use random selection
        if self.model is None:
            print("WARNING: Model is None, using random move selection")
            print("This could be due to:")
            print("1. TensorFlow or Keras not being available")
            print("2. Model file not being found")
            print("3. Error during model loading")
            print("Check the logs above for more details")
            # Shuffle the valid moves to get a random one
            random.shuffle(valid_moves)

            # Try each move until we find a valid one
            for selected_move in valid_moves:
                # Validate the selected move
                if isinstance(selected_move, Move) and isinstance(selected_move.piece, Piece):
                    # Double-check that the move is valid
                    piece = selected_move.piece
                    if board.valid_move(piece, selected_move):
                        print(f"Selected valid move: {selected_move}")
                        return selected_move

            # If we couldn't find a valid move, try to create one
            print("No valid moves found in the pre-calculated list, trying to create a new move")
            for row in range(8):
                for col in range(8):
                    piece = board.squares[row][col].piece
                    if piece and piece.color == 'black':
                        # Clear previous moves
                        piece.clear_moves()
                        # Calculate new moves
                        board.calc_moves(piece, row, col, bool=True)
                        # Find a valid move for this piece
                        if piece.moves:
                            selected_move = piece.moves[0]
                            # Ensure the move has a valid piece
                            if not isinstance(selected_move, Move) or not isinstance(selected_move.piece, Piece):
                                print(f"Created move is invalid: {selected_move}")
                                # Try to create a valid move manually
                                from square import Square
                                initial = Square(row, col, piece)
                                final = Square(selected_move.final.row, selected_move.final.col)
                                selected_move = Move(initial, final, piece)
                            print(f"Created new move: {selected_move}")
                            return selected_move

            # If we still can't find a valid move, assume checkmate
            print("Could not find or create a valid move. Assuming checkmate.")
            return None

        try:
            print("Starting AI prediction process...")
            # Convert the board state to an input suitable for the AI model
            print("Converting board state to model input...")
            board_state = self.convert_board_to_input(board)

            # Debugging: Output the board state shape
            print(f"Board state shape: {board_state.shape}")

            # Check if model has predict method
            if not hasattr(self.model, 'predict'):
                print("ERROR: Model does not have predict method!")
                raise AttributeError("Model does not have predict method")

            # Try to get prediction from model
            print("Calling model.predict with board state...")
            try:
                prediction = self.model.predict(board_state)
                print("Model prediction successful")
            except Exception as pred_error:
                print(f"Error during model prediction: {pred_error}")
                # Try with a different approach if available
                if hasattr(self.model, '__call__'):
                    print("Trying model.__call__ as an alternative to predict...")
                    try:
                        prediction = self.model(board_state)
                        print("Model call successful")
                    except Exception as call_error:
                        print(f"Error during model call: {call_error}")
                        # Try with a different input format
                        print("Trying with different input format...")
                        try:
                            # Try with a list instead of numpy array
                            prediction = self.model(board_state.tolist())
                            print("Model call with list input successful")
                        except Exception as list_error:
                            print(f"Error during model call with list input: {list_error}")
                            # Try with a tensor if TensorFlow is available
                            if TF_AVAILABLE:
                                try:
                                    print("Trying with TensorFlow tensor...")
                                    tensor_input = tf.convert_to_tensor(board_state)
                                    prediction = self.model(tensor_input)
                                    print("Model call with tensor input successful")
                                except Exception as tensor_error:
                                    print(f"Error during model call with tensor input: {tensor_error}")
                                    raise pred_error
                            else:
                                raise pred_error
                else:
                    raise pred_error

            # Debugging: Output the prediction shape and content
            print(f"Prediction shape: {prediction.shape if hasattr(prediction, 'shape') else 'no shape attribute'}")
            print(f"Prediction content: {prediction}")
            print(f"Prediction type: {type(prediction)}")

            # Try to normalize the prediction format
            print("Normalizing prediction format...")
            try:
                # Convert to numpy array if it's not already
                if not isinstance(prediction, np.ndarray):
                    if TF_AVAILABLE and hasattr(tf, 'Tensor') and isinstance(prediction, tf.Tensor):
                        print("Converting TensorFlow tensor to numpy array...")
                        prediction = prediction.numpy()
                    else:
                        print("Converting to numpy array...")
                        prediction = np.array(prediction)

                # Ensure it's at least 2D (batch, predictions)
                if len(prediction.shape) == 1:
                    print("Expanding dimensions to make it 2D...")
                    prediction = np.expand_dims(prediction, axis=0)

                print(f"Normalized prediction shape: {prediction.shape}")
            except Exception as norm_error:
                print(f"Error during prediction normalization: {norm_error}")
                # Continue with the original prediction format

            # Select the best move based on the model's prediction
            print(f"Selecting best move based on prediction (valid moves: {len(valid_moves)})...")
            selected_move = self.select_best_move(prediction, valid_moves)

            # Validate the selected move
            if isinstance(selected_move, Move) and isinstance(selected_move.piece, Piece):
                # Double-check that the move is valid
                piece = selected_move.piece
                if board.valid_move(piece, selected_move):
                    print(f"Selected valid move from model: {selected_move}")
                    return selected_move
                else:
                    print(f"Warning: Model selected an invalid move, using random move instead")
            else:
                print(f"Warning: Invalid move object from model, using random move instead")
                # Try to find a valid move with the same coordinates
                if hasattr(selected_move, 'initial') and hasattr(selected_move, 'final'):
                    for move in valid_moves:
                        if (move.initial.row == selected_move.initial.row and 
                            move.initial.col == selected_move.initial.col and 
                            move.final.row == selected_move.final.row and 
                            move.final.col == selected_move.final.col and
                            isinstance(move, Move) and 
                            isinstance(move.piece, Piece)):
                            print(f"Found valid move with same coordinates: {move}")
                            return move

            # Fall back to random selection if model selection fails
            random.shuffle(valid_moves)
            for move in valid_moves:
                if isinstance(move, Move) and isinstance(move.piece, Piece) and board.valid_move(move.piece, move):
                    return move

            # If we still can't find a valid move, assume checkmate
            print("Could not find a valid move after model prediction. Assuming checkmate.")
            return None

        except Exception as e:
            print(f"ERROR during AI move selection: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            print("Falling back to random move selection")

            # Try each move until we find a valid one
            print(f"Shuffling {len(valid_moves)} valid moves for random selection...")
            random.shuffle(valid_moves)
            for i, move in enumerate(valid_moves):
                print(f"Checking random move {i+1}/{len(valid_moves)}: {move}")
                if isinstance(move, Move) and isinstance(move.piece, Piece):
                    print(f"Move and piece are valid instances")
                    if board.valid_move(move.piece, move):
                        print(f"Move is valid according to board.valid_move")
                        print(f"Selected random move: {move}")
                        return move
                    else:
                        print(f"Move is not valid according to board.valid_move")
                else:
                    print(f"Move or piece is not a valid instance. Move type: {type(move)}, " + 
                          f"Piece type: {type(move.piece) if hasattr(move, 'piece') else 'no piece attribute'}")

            # If we still can't find a valid move, assume checkmate
            print("Could not find a valid move after exception. Assuming checkmate.")
            return None

    def convert_board_to_input(self, board):
        """Convert the board representation to a model-friendly format."""
        board_state = np.zeros((8, 8, 12))  # Assuming 12 piece types (6 per color)
        for row in range(8):
            for col in range(8):
                piece = board.squares[row][col].piece
                if piece:
                    piece_type = self.get_piece_channel(piece)
                    board_state[row][col][piece_type] = 1
        return np.expand_dims(board_state, axis=0)

    def get_piece_channel(self, piece):
        """Map each piece to its respective channel."""
        piece_map = {
            "pawn": 0, "rook": 1, "knight": 2,
            "bishop": 3, "queen": 4, "king": 5
        }
        piece_type = piece.name
        if piece_type not in piece_map:
            raise ValueError(f"Unknown piece type: {piece_type}")
        channel = piece_map[piece_type]
        if piece.color == "black":
            channel += 6  # Offset to differentiate black pieces
        return channel

    def get_valid_moves(self, board):
        """Retrieve all valid moves for the AI's pieces."""
        valid_moves = []
        for row in range(8):
            for col in range(8):
                piece = board.squares[row][col].piece
                if piece and piece.color == 'black':  # AI plays black
                    # Clear previous moves
                    piece.clear_moves()
                    # Calculate new moves
                    board.calc_moves(piece, row, col, bool=True)
                    # Add valid moves to the list
                    for move in piece.moves:
                        if board.valid_move(piece, move):
                            valid_moves.append(move)

        # Debug output
        print(f"Found {len(valid_moves)} valid moves for AI")

        if not valid_moves:
            raise ValueError("No valid moves found for the AI!")
        return valid_moves

# fix "Mismatch between predictions and valid moves count"

    def select_best_move(self, prediction, valid_moves):
        """
        Associate model predictions with valid moves and select the best move.
        """
        print("Entering select_best_move method...")

        # Check if there are no valid moves
        if not valid_moves:
            print("ERROR: No valid moves available for selection by AI!")
            raise RuntimeError("No valid moves available for selection by AI!")

        # Check prediction type and structure
        print(f"Prediction type: {type(prediction)}")
        if isinstance(prediction, (list, tuple)):
            print(f"Prediction is a {type(prediction).__name__} with length {len(prediction)}")
            if len(prediction) > 0:
                print(f"First element type: {type(prediction[0])}")
                if hasattr(prediction[0], 'shape'):
                    print(f"First element shape: {prediction[0].shape}")
        elif hasattr(prediction, 'shape'):
            print(f"Prediction shape: {prediction.shape}")

        # Try to get the prediction values in a usable format
        try:
            if isinstance(prediction, (list, tuple)) and len(prediction) > 0:
                pred_values = prediction[0]
            elif hasattr(prediction, 'shape') and len(prediction.shape) > 1:
                pred_values = prediction[0]
            else:
                pred_values = prediction

            print(f"Prediction values type: {type(pred_values)}")
            if hasattr(pred_values, 'shape'):
                print(f"Prediction values shape: {pred_values.shape}")
            elif hasattr(pred_values, '__len__'):
                print(f"Prediction values length: {len(pred_values)}")

            # Check if the prediction length matches the number of valid moves
            pred_len = len(pred_values) if hasattr(pred_values, '__len__') else 0
            print(f"Prediction length: {pred_len}, Valid moves count: {len(valid_moves)}")

            if pred_len != len(valid_moves):
                print(f"WARNING: Model prediction output ({pred_len}) does not match the "
                      f"number of valid moves ({len(valid_moves)}).")

                # Instead of using random selection, try to adapt the prediction to the valid moves
                if pred_len > 0:
                    print("Attempting to adapt prediction to valid moves...")

                    # If we have more predictions than moves, we can just use the first len(valid_moves) predictions
                    if pred_len > len(valid_moves):
                        print(f"More predictions than moves, using first {len(valid_moves)} predictions")
                        # We'll handle this in the move_scores creation below

                    # If we have fewer predictions than moves, we need to pad the predictions
                    elif pred_len < len(valid_moves):
                        print(f"Fewer predictions than moves, padding with zeros")
                        # We'll handle this in the move_scores creation below

                    # Continue with the adapted prediction
                    print("Continuing with adapted prediction")
                else:
                    print("No usable predictions, using random selection")
                    # If there's no usable prediction, just select a random move
                    random_move = random.choice(valid_moves)
                    print(f"Randomly selected move: {random_move}")
                    return random_move

            # Pair valid moves with prediction scores
            print("Pairing valid moves with prediction scores...")
            move_scores = {}

            # Handle the case where prediction length doesn't match valid moves count
            if pred_len != len(valid_moves):
                # If we have more predictions than moves, use only the first len(valid_moves) predictions
                if pred_len > len(valid_moves):
                    for idx, move in enumerate(valid_moves):
                        score = float(pred_values[idx])  # Convert to float to ensure it's a scalar
                        move_scores[move] = score
                        print(f"Move {idx}: {move}, Score: {score}")

                # If we have fewer predictions than moves, pad with zeros or use available predictions
                elif pred_len < len(valid_moves):
                    # First, use the available predictions
                    for idx in range(pred_len):
                        move = valid_moves[idx]
                        score = float(pred_values[idx])  # Convert to float to ensure it's a scalar
                        move_scores[move] = score
                        print(f"Move {idx}: {move}, Score: {score}")

                    # Then, pad with small random values for the remaining moves
                    # Using small random values instead of zeros to avoid ties
                    for idx in range(pred_len, len(valid_moves)):
                        move = valid_moves[idx]
                        # Use a small random value (0.01-0.1) to avoid ties but still prioritize predicted moves
                        score = random.uniform(0.01, 0.1)
                        move_scores[move] = score
                        print(f"Move {idx}: {move}, Score: {score} (padded)")
            else:
                # Normal case: prediction length matches valid moves count
                for idx, move in enumerate(valid_moves):
                    score = float(pred_values[idx])  # Convert to float to ensure it's a scalar
                    move_scores[move] = score
                    print(f"Move {idx}: {move}, Score: {score}")

            if not move_scores:
                print("ERROR: No move scores were created. Using random selection.")
                random_move = random.choice(valid_moves)
                print(f"Randomly selected move: {random_move}")
                return random_move

            # Select the move with the highest associated prediction score
            print("Selecting move with highest score...")
            best_move = max(move_scores, key=move_scores.get)
            best_score = move_scores[best_move]
            print(f"Best move: {best_move}, Score: {best_score}")

            # Validate the selected move
            if best_move not in valid_moves:
                print(f"ERROR: Selected move {best_move} is not in the list of valid moves!")
                raise RuntimeError(f"Selected move {best_move} is not in the list of valid moves!")

            if not isinstance(best_move, Move) or not isinstance(best_move.piece, Piece):
                print(f"WARNING: Invalid move or piece detected in: {best_move}")
                # Try to find a valid move with the same coordinates
                for move in valid_moves:
                    if (move.initial.row == best_move.initial.row and 
                        move.initial.col == best_move.initial.col and 
                        move.final.row == best_move.final.row and 
                        move.final.col == best_move.final.col and
                        isinstance(move, Move) and 
                        isinstance(move.piece, Piece)):
                        print(f"Found valid move with same coordinates: {move}")
                        return move
                # If we can't find a valid move with the same coordinates, raise an error
                print(f"ERROR: Invalid move or piece detected in: {best_move} and no valid alternative found")
                raise ValueError(f"Invalid move or piece detected in: {best_move}")

            print(f"Returning best move: {best_move}")
            return best_move

        except Exception as e:
            print(f"ERROR in select_best_move: {e}")
            print("Using random selection due to error")
            random_move = random.choice(valid_moves)
            print(f"Randomly selected move: {random_move}")
            return random_move
