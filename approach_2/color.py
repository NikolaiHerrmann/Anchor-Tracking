
from enum import Enum
import numpy as np


class Color(Enum):

    BLUE = 0
    BROWN = 1
    YELLOW = 2
    YELLOW1 = 7
    GREEN1 = 3
    PINK = 4
    GREEN = 5
    ORANGE = 6
    BROWN1 = 8
    GREEN2= 99

    def hsv(self):
        return Color.lookup.get(self)[0]

    def bgr(self):
        return Color.lookup.get(self)[1]

    def name(self):
        return Color.lookup.get(self)[2]

Color.lookup = {
    Color.YELLOW1: (np.array([[0, 89, 148], [64, 180, 215]]), (0, 255, 255), "yellow1"),
    Color.YELLOW: (np.array([[14, 85, 138], [26, 141, 255]]), (0, 255, 255), "yellow"),
    Color.GREEN1: (np.array([[58, 144, 30], [100, 255, 77]]), (0, 255, 0), "green1"),
    Color.GREEN: (np.array([[65, 117, 47], [96, 255, 95]]), (0, 255, 0), "green"),
    Color.BLUE: (np.array([[94, 120, 87], [154, 255, 255]]), (255, 0, 0), "blue"),
    Color.ORANGE: (np.array([[0, 88, 141], [10, 179, 185]]), (0, 98, 179), "orange"),
    Color.BROWN: (np.array([[0, 79, 83], [10, 146, 123]]), (0, 0, 255), "brown"),
    Color.BROWN1: (np.array([[0, 46, 63], [14, 148, 121]]), (42, 42, 165), "brown1"),
    Color.PINK: (np.array([[162, 148,  62], [175, 238, 174]]), (203, 192, 255), "pink"),
    Color.GREEN2: (np.array([[57, 62, 60 ], [91, 169, 128]]), (0, 255, 0), "green2")
}

# Color.lookup = {
#     Color.GREEN: (np.array([[58, 144, 30], [100, 255, 77]]), (0, 255, 0), "green"),
#     #Color.GREEN: (np.array([[70, 73, 82], [84, 255, 134]]), (0, 255, 0), "green"),
#     Color.BLUE: (np.array([[93, 135, 109], [105, 232, 223]]), (255, 0, 0), "blue"),
#     Color.BROWN: (np.array([[0, 46, 63], [14, 148, 121]]), (42, 42, 165), "brown"),
#     Color.YELLOW: (np.array([[0, 89, 148], [64, 180, 215]]), (0, 255, 255), "yellow"),
#     Color.PINK: (np.array([[162, 148,  62], [175, 238, 174]]), (203, 192, 255), "pink")
# }
