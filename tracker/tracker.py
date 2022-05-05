
import cv2
import os
from object import Object
from color import Color
from anchor import Anchor


class Tracker:

    ESC = 27
    SCREEN_MAIN = "Object Tracker"
    EXIT_DELAY = 1
    ACCURACY_LOC = (50, 50)

    CAMERA_RES = (640, 480)
    CAMERA_FPS = 30

    RECORD_FORMAT = 'MJPG'
    RECORD_NAME = "track_video_5t.mp4"
    DATA_PATH = "data"

    def __init__(self, camera_arg, is_training):
        self.is_training = is_training

        self.vid_cap = cv2.VideoCapture(camera_arg)
        self.vid_cap.set(cv2.CAP_PROP_FRAME_WIDTH, Tracker.CAMERA_RES[0])
        self.vid_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Tracker.CAMERA_RES[1])
        
        self.objects = []
        self.anchor = Anchor(self.is_training)

        for color in Color:
            self.objects.append(Object(color, self.is_training))

        if self.is_training:
            path = os.path.join("..", Tracker.DATA_PATH)
            if not os.path.isdir(path):
                os.mkdir(path)
            path = os.path.join(path, Tracker.RECORD_NAME)
            fourcc = cv2.VideoWriter_fourcc(*Tracker.RECORD_FORMAT)
            self.out_stream = cv2.VideoWriter(path, fourcc, Tracker.CAMERA_FPS, Tracker.CAMERA_RES)

    def run(self):
        while True:
            ret, frame = self.vid_cap.read()

            if not ret:
                break

            if self.is_training:
                self.out_stream.write(frame)

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            for obj in self.objects:
                found, frame = obj.draw(hsv_frame, frame)
                if found:
                    self.anchor.match(obj)

            if not self.is_training:
                text = "Accuracy=" + str(round(self.anchor.get_accuracy(), 3))
                cv2.putText(frame, text, Tracker.ACCURACY_LOC, cv2.FONT_HERSHEY_PLAIN, Object.TEXT_SCALE, Object.RGB_WHITE)

            cv2.imshow(Tracker.SCREEN_MAIN, frame)

            if cv2.waitKey(Tracker.EXIT_DELAY) == Tracker.ESC:
                break

        self.anchor.save_data()

        self.vid_cap.release()
        if self.is_training:
            self.out_stream.release()

        cv2.destroyAllWindows()
