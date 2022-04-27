
import cv2
from object import Object
from color import Color
from anchor import Anchor


class Tracker:

    ESC = 27
    SCREEN_MAIN = "Object Tracker"
    EXIT_DELAY = 1

    FPS = 30
    RECORD_RES = (640, 480)

    def __init__(self, camera_idx, record=False):
        self.vid_cap = cv2.VideoCapture(camera_idx)
        self.record = record
        self.objects = []
        self.anchor = Anchor()

        for color in Color:
            self.objects.append(Object(color))

        if record:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out_stream = cv2.VideoWriter('output.avi', fourcc, Tracker.FPS, Tracker.RECORD_RES)

    def run(self):
        while True:
            _, frame = self.vid_cap.read()

            if self.record:
                self.out_stream.write(frame)

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            for obj in self.objects:
                found, frame = obj.draw(hsv_frame, frame)
                if found:
                    self.anchor.match(obj)

            cv2.imshow(Tracker.SCREEN_MAIN, frame)

            if cv2.waitKey(Tracker.EXIT_DELAY) == Tracker.ESC:
                break

        self.anchor.save_data()
        self.vid_cap.release()
        if self.record:
            self.out_stream.release()
        cv2.destroyAllWindows()
