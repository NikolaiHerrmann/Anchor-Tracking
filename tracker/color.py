
from enum import Enum
import numpy as np


class Color(Enum):

    GREEN = 0
    #BLUE = 1
    RED = 1
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
    Color.GREEN: np.array([[66, 109, 112], [86, 164, 185]]),
    #Color.BLUE: np.array([[97, 35, 186], [104, 143, 223]]),
    Color.RED: np.array([[0, 115, 170], [179, 228, 228]])
}
