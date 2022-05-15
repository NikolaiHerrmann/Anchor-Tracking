
from enum import Enum
import numpy as np


class Color(Enum):

    #
    BLUE = 0
    #BROWN = 1
    #YELLOW = 2
    #GREEN = 3
    #BLACK = 0
    # ORANGE = 4
    # WHITE = 5
    # BLACK = 6
    # PURPLE = 7
    # BROWN = 8

    def get_hsv_bounds(self):
        return Color._lookup.get(self)

    def update_hsv_bounds(color, bounds):
        Color._lookup[color] = bounds


Color._lookup = {
    #Color.GREEN: np.array([[58, 144, 30], [100, 255, 77]]),
    Color.BLUE: np.array([[93, 135, 109], [105, 232, 223]]),
    #Color.BROWN: np.array([[0, 46, 63], [14, 148, 121]]),
    #Color.YELLOW: np.array([[0, 89, 148], [64, 180, 215]]),
    # Color.BLACK: np.array([[0, 0, 0], [0, 0, 0]])
}
