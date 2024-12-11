import tensorflow as tf
import numpy as np

from board import Board


class AIPlayer:
    def __init__(self):
        self.model = tf.keras.models.load_model('assets/chessModel.keras')

    def get_move(self, board):
        # Convert current board state to model input format
        board_state = self.convert_board_to_input(board)

        # Get model prediction
        prediction = self.model.predict(board_state)

        # Convert prediction to valid move
        valid_moves = self.get_valid_moves(board)
        if not valid_moves:
            raise ValueError("No valid moves available!")
        chosen_move = self.select_best_move(prediction, valid_moves)

        return chosen_move

    def convert_board_to_input(self, board):
        # Convert board state to match training data format (8x8x12)
        board_state = np.zeros((8, 8, 12))  # Assuming 12 piece types

        for row in range(8):
            for col in range(8):
                piece = board.squares[row][col].piece
                if piece:
                    # Map piece to correct channel based on your training data
                    piece_type = self.get_piece_channel(piece)
                    board_state[row][col][piece_type] = 1

        return np.expand_dims(board_state, axis=0)  # Add batch dimension

    def get_piece_channel(self, piece):
        # Map pieces to training channels (one-hot encoding for 12 channels: 6 white, 6 black)
        piece_map = {
            "pawn": 0, "rook": 1, "knight": 2,
            "bishop": 3, "queen": 4, "king": 5
        }

        # Use the 'name' attribute instead of 'type'
        piece_type = piece.name

        if piece_type not in piece_map:
            raise ValueError(f"Unknown piece type: {piece_type}")

        channel = piece_map[piece_type]  # Map to channel
        if piece.color == "black":
            channel += 6  # Offset by 6 if the piece is black
        return channel

    def get_valid_moves(self, board):
        # Get all valid moves for black pieces
        valid_moves = []
        for row in range(8):
            for col in range(8):
                piece = board.squares[row][col].piece
                if piece and piece.color == 'black':
                    board.calc_moves(piece, row, col, bool=True)
                    valid_moves.extend(piece.moves)
        return valid_moves

    def select_best_move(self, prediction, valid_moves):
        # Convert model prediction to actual move
        # This depends on your model's output format
        move_scores = {}
        for idx, move in enumerate(valid_moves):
            move_scores[move] = prediction[0][idx]  # Model output assumed to be probabilities

        # Pick the move with the highest predicted score
        return max(move_scores, key=move_scores.get)
