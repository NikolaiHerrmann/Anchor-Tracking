
import enum
import numpy as np


class Color(enum.Enum):
    
    GREEN = 1
    BLUE = 2

    def get_hsv_bounds(self):
        if self == Color.GREEN:
            return np.array([[90, 30, 244], [102, 150, 255]])
        if self == Color.BLUE:
            return np.array([[78, 104, 114], [83, 226, 239]])
