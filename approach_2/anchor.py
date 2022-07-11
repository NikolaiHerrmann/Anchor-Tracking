
import numpy as np
import cv2

class Anchor:

    MAX_ID = 0

    def __init__(self, obj, t):
        self.id = Anchor.MAX_ID
        Anchor.MAX_ID += 1
        self.update(obj, t)

    def update(self, obj, t):
        self.position = obj.get_position()
        self.size = obj.get_size()
        self.t = t
        self.track_color = obj.get_track_color()

    def get_position_thresh(self, obj):
        l2 = np.linalg.norm(self.position - obj.get_position())
        # return np.power(np.e, -l2)
        return l2

    def get_size_thresh(self, obj):
        other_size = obj.get_size()
        # return cv2.matchShapes(self.contour_mask, other_mask, cv2.CONTOURS_MATCH_I1, 0.0)
        return np.abs(self.size - other_size)
        
    def get_time_thresh(self, t):
        # return 2 / (1 + np.power(np.e, (t - self.t)))
        return t - self.t

    def get_target_class(self, obj):
        return 1 if self.track_color == obj.get_track_color() else 0

    def track_update(self, obj, t):
        self.position = obj.get_track_position()
        self.t = t

    def get_id(self):
        return self.id

    def reset():
        Anchor.MAX_ID = 0