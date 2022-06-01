
from enum import Enum
import numpy as np


class Color(Enum):

    BLUE = 0
    BROWN = 1
    YELLOW = 2
    GREEN = 3
    PINK = 4

    def hsv(self):
        return Color.lookup.get(self)[0]

    def bgr(self):
        return Color.lookup.get(self)[1]

    def name(self):
        return Color.lookup.get(self)[2]


Color.lookup = {
    Color.GREEN: (np.array([[58, 144, 30], [100, 255, 77]]), (0, 255, 0), "green"),
    Color.BLUE: (np.array([[93, 135, 109], [105, 232, 223]]), (255, 0, 0), "blue"),
    Color.BROWN: (np.array([[0, 46, 63], [14, 148, 121]]), (42, 42, 165), "brown"),
    Color.YELLOW: (np.array([[0, 89, 148], [64, 180, 215]]), (0, 255, 255), "yellow"),
    Color.PINK: (np.array([[162, 148,  62], [175, 238, 174]]), (203, 192, 255), "pink")
}
