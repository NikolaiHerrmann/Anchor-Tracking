
import cv2
from object import Object
from color import Color
from anchor import Anchor


class Tracker:

    ESC = 27
    SCREEN_MAIN = "Object Tracker"
    EXIT_DELAY = 1

    def __init__(self, vid_cap):
        self.vid_cap = vid_cap
        self.objects = []
        self.anchor = Anchor()

        for color in Color:
            self.objects.append(Object(color))

    def run(self):
        while True:
            _, frame = self.vid_cap.read()
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            for obj in self.objects:
                found, frame = obj.draw(hsv_frame, frame)
                if found:
                    self.anchor.match_generate_data(obj)

            cv2.imshow(Tracker.SCREEN_MAIN, frame)

            if cv2.waitKey(Tracker.EXIT_DELAY) == Tracker.ESC:
                self.anchor.save_data()
                break
