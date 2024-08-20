from colour import Colour
class Theme:

    def __init__(self, lightBG, darkBG
                     , lightTrace, darkTrace
                     , lightMoves, darkMoves):

        self.bg = Colour(lightBG, darkBG)
        self.trace = Colour(lightTrace, darkTrace)
        self.moves = Colour(lightMoves, darkMoves)