class Move:

    def __init__(self, initial, final, piece=None):
        # initial and final are squares
        self.initial = initial
        self.final = final
        self.piece = piece
        self.piece_captured = None

    def __repr__(self):
        return f"Move({self.initial}, {self.final}, {self.piece})"

    def __hash__(self):
        # Combines start and end positions to create a unique hash
        return hash((self.initial, self.final))

    def __str__(self):
        s = ''
        s += f'({self.initial.col}, {self.initial.row})'
        s += f' -> ({self.final.col}, {self.final.row})'
        return s

    def __eq__(self, other):
        return self.initial == other.initial and self.final == other.final
