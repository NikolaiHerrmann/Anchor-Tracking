
from enum import Enum
import numpy as np


class Color(Enum):

    GREEN = 0
    BLUE = 1
    RED = 2
    # YELLOW = 3
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
    Color.GREEN: np.array([[78, 71, 178], [84, 160, 255]]),
    Color.BLUE: np.array([[85, 33, 236], [104, 130, 255]]),
    Color.RED: np.array([[0, 140, 247], [14, 202, 255]])
}
