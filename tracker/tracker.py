
import cv2
import os
from object import Object
from color import Color
from anchor_manager import AnchorManager
import time


class Tracker:

    SCREEN_MAIN = "Object Tracker"
    EXIT_DELAY = 1

    CAMERA_RES = (640, 480)
    CAMERA_FPS = 30

    RECORD_FORMAT = 'MJPG'
    RECORD_NAME = "ml_tracking.mp4"
    DATA_PATH = "data"

    ESC = 27

    def __init__(self, camera_arg, is_training, kalman_help):
        self.is_training = is_training

        self.vid_cap = cv2.VideoCapture(camera_arg)
        if not self.vid_cap.isOpened():
            raise Exception("Video or camera failed to open!")

        self.vid_cap.set(cv2.CAP_PROP_FRAME_WIDTH, Tracker.CAMERA_RES[0])
        self.vid_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Tracker.CAMERA_RES[1])
        
        self.objects = []
        self.anchor_manager = AnchorManager(self.is_training, kalman_help)

        for color in Color:
            self.objects.append(Object(color, self.is_training))

        self.is_training = True

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
            #time.sleep(0.1)
            
            #self.out_stream.write(frame)

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            for obj in self.objects:
                found, frame = obj.detect(hsv_frame, frame)
                self.anchor_manager.match(obj, frame, found)

            cv2.imshow(Tracker.SCREEN_MAIN, frame)

            if self.is_training:
                self.out_stream.write(frame)

            if cv2.waitKey(Tracker.EXIT_DELAY) == Tracker.ESC:
                break

        self.anchor_manager.save_data()

        self.vid_cap.release()
        if self.is_training:
            self.out_stream.release()

        cv2.destroyAllWindows()
