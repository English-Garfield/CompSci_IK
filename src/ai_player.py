import tensorflow as tf
import numpy as np

from board import Board
from move import Move
from piece import Piece


class AIPlayer:
    def __init__(self):
        self.model = tf.keras.models.load_model('assets/chessModel.keras')
        self.board = Board()

    def get_move(self, board):
        # Convert the board state to an input suitable for the AI model
        board_state = self.convert_board_to_input(board)

        # Debugging: Output the board state shape
        print(f"Board state shape: {board_state.shape}")

        prediction = self.model.predict(board_state)
        # Debugging: Output the prediction shape and content
        print(f"Prediction shape: {prediction.shape}")
        print(f"Prediction content: {prediction}")

        # Get valid moves for the AI's pieces
        valid_moves = self.get_valid_moves(board)

        # Debugging: Output the number of valid moves
        print(f"Number of valid moves: {len(valid_moves)}")

        if not valid_moves:
            raise ValueError("No valid moves available!")

        # Select the best move based on the model's prediction
        try:
            selected_move = self.select_best_move(prediction, valid_moves)
        except Exception as e:
            raise RuntimeError(f"Failed to select valid move: {e}")

        # Validate the selected move
        if isinstance(selected_move, Move) and isinstance(selected_move.piece, Piece):
            return selected_move
        else:
            raise ValueError(f"Invalid move or piece detected in: {selected_move}")

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
                    board.calc_moves(piece, row, col, bool=True)
                    valid_moves.extend(piece.moves)
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
            raise ValueError(
                f"Model prediction output ({len(prediction[0])}) does not match the "
                f"number of valid moves ({len(valid_moves)})."
            )

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