
import cv2
from object import Object
from color import Color
from anchor import Anchor


class Tracker:

    ESC = 27
    SCREEN_MAIN = "Object Tracker"
    EXIT_DELAY = 1
    ACCURACY_LOC = (50, 50)
    FPS = 30
    RECORD_RES = (640, 480)

    IS_TRAINING = False

    def __init__(self, camera_idx, record=False):
        self.vid_cap = cv2.VideoCapture(camera_idx)
        self.record = record
        self.objects = []
        self.anchor = Anchor(is_training=Tracker.IS_TRAINING)

        for color in Color:
            self.objects.append(Object(color))

        if record:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out_stream = cv2.VideoWriter('output.avi', fourcc, Tracker.FPS, Tracker.RECORD_RES)

    def run(self):
        while True:
            _, frame = self.vid_cap.read()

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            for obj in self.objects:
                found, frame = obj.draw(hsv_frame, frame)
                if found:
                    self.anchor.match(obj)

            if not Tracker.IS_TRAINING:
                text = "Accuracy=" + str(round(self.anchor.get_accuracy(), 3))
                cv2.putText(frame, text, Tracker.ACCURACY_LOC, cv2.FONT_HERSHEY_PLAIN, Object.TEXT_SCALE, Object.RGB_WHITE)

            cv2.imshow(Tracker.SCREEN_MAIN, frame)

            if self.record:
                self.out_stream.write(frame)

            if cv2.waitKey(Tracker.EXIT_DELAY) == Tracker.ESC:
                break

        self.anchor.save_data()
        self.vid_cap.release()
        if self.record:
            self.out_stream.release()
        cv2.destroyAllWindows()
