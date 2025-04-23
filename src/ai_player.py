import numpy as np
import os
import random

# Try different import strategies
try:
    import keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("Warning: Keras not available, will use random moves")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available, will use random moves")


from board import Board
from move import Move
from piece import Piece


class AIPlayer:
    def __init__(self):
        self.board = Board()
        self.model = None

        # Try different methods to load the model
        model_path = 'assets/chessModel.keras'
        if os.path.exists(model_path):
            if KERAS_AVAILABLE:
                try:
                    self.model = keras.models.load_model(model_path)
                    print("Model loaded successfully with keras")
                except Exception as e:
                    print(f"Failed to load model with keras: {e}")

            if self.model is None and TF_AVAILABLE:
                try:
                    self.model = tf.keras.models.load_model(model_path)
                    print("Model loaded successfully with tensorflow")
                except Exception as e:
                    print(f"Failed to load model with tensorflow: {e}")

            if self.model is None:
                print("Warning: Could not load AI model, will use random moves")

    def get_move(self, board):
        # Get valid moves for the AI's pieces
        valid_moves = self.get_valid_moves(board)

        # Debugging: Output the number of valid moves
        print(f"Number of valid moves: {len(valid_moves)}")

        if not valid_moves:
            raise ValueError("No valid moves available!")

        # If model is not available, use random selection
        if self.model is None:
            print("Using random move selection (model not available)")
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
                            print(f"Created new move: {selected_move}")
                            return selected_move

            # If we still can't find a valid move, raise an error
            raise ValueError("Could not find or create a valid move")

        try:
            # Convert the board state to an input suitable for the AI model
            board_state = self.convert_board_to_input(board)

            # Debugging: Output the board state shape
            print(f"Board state shape: {board_state.shape}")

            # Try to get prediction from model
            prediction = self.model.predict(board_state)

            # Debugging: Output the prediction shape and content
            print(f"Prediction shape: {prediction.shape}")
            print(f"Prediction content: {prediction}")

            # Select the best move based on the model's prediction
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

            # Fall back to random selection if model selection fails
            random.shuffle(valid_moves)
            for move in valid_moves:
                if isinstance(move, Move) and isinstance(move.piece, Piece) and board.valid_move(move.piece, move):
                    return move

            # If we still can't find a valid move, raise an error
            raise ValueError("Could not find a valid move after model prediction")

        except Exception as e:
            print(f"Error during AI move selection: {e}")
            print("Falling back to random move selection")

            # Try each move until we find a valid one
            random.shuffle(valid_moves)
            for move in valid_moves:
                if isinstance(move, Move) and isinstance(move.piece, Piece) and board.valid_move(move.piece, move):
                    return move

            # If we still can't find a valid move, raise an error
            raise ValueError("Could not find a valid move after exception")

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

# Python code to fix "Mismatch between predictions and valid moves count"

    def select_best_move(self, prediction, valid_moves):
        """
        Associate model predictions with valid moves and select the best move.
        """
        # Check if there are no valid moves
        if not valid_moves:
            raise RuntimeError("No valid moves available for selection by AI!")

        # Check if the prediction length matches the number of valid moves
        if len(prediction[0]) != len(valid_moves):
            print(f"Warning: Model prediction output ({len(prediction[0])}) does not match the "
                  f"number of valid moves ({len(valid_moves)}). Using random selection.")
            # If there's a mismatch, just select a random move
            import random
            return random.choice(valid_moves)

        # Pair valid moves with prediction scores
        move_scores = {move: prediction[0][idx] for idx, move in enumerate(valid_moves)}

        # Debugging: Output moves and their scores
        for move, score in move_scores.items():
            print(f"Move: {move}, Score: {score}")

        # Select the move with the highest associated prediction score
        best_move = max(move_scores, key=move_scores.get)

        # Validate the selected move
        if best_move not in valid_moves:
            raise RuntimeError(f"Selected move {best_move} is not in the list of valid moves!")

        if not isinstance(best_move, Move) or not isinstance(best_move.piece, Piece):
            raise ValueError(f"Invalid move or piece detected in: {best_move}")

        return best_move
