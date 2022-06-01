
import cv2
import os
from object import Object
from color import Color
from anchor_manager import AnchorManager


class Tracker:

    FRAME_TITLE = "Object Tracker"
    FRAME_DISPLAY_TIME = 1 # ms

    CAMERA_RES = (640, 480)
    CAMERA_FPS = 30

    RECORD_FORMAT = 'MJPG'
    RECORD_NAME = "ml_tracking_improved.mp4"
    
    ESC = 27

    def __init__(self, camera_arg, data_path, is_training):
        self.is_training = is_training

        self.vid_cap = cv2.VideoCapture(camera_arg)
        if not self.vid_cap.isOpened():
            raise Exception("Video file or camera failed to open!")

        self.vid_cap.set(cv2.CAP_PROP_FRAME_WIDTH, Tracker.CAMERA_RES[0])
        self.vid_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Tracker.CAMERA_RES[1])

        self.objects = []
        self.anchor_manager = AnchorManager(self.is_training)

        for color in Color:
            self.objects.append(Object(color, self.is_training))
        
        if not os.path.isdir(data_path):
            os.mkdir(data_path)
        fourcc = cv2.VideoWriter_fourcc(*Tracker.RECORD_FORMAT)
        self.out_stream = cv2.VideoWriter(os.path.join(data_path, Tracker.RECORD_NAME),
                                          fourcc, Tracker.CAMERA_FPS, Tracker.CAMERA_RES)

    def run(self):
        while True:
            ret, frame = self.vid_cap.read()

            if not ret:
                break

            if self.is_training:
                self.out_stream.write(frame)

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            self.anchor_manager.reset_ids()
            
            for obj in self.objects:
                found, frame = obj.detect(hsv_frame, frame)
                self.anchor_manager.match(obj, frame, found)

            cv2.imshow(Tracker.FRAME_TITLE, frame)

            if not self.is_training:
                self.out_stream.write(frame)

            if cv2.waitKey(Tracker.FRAME_DISPLAY_TIME) == Tracker.ESC:
                break
        
        self.anchor_manager.save_data()

    def close(self):
        self.out_stream.release()
        self.vid_cap.release()
        cv2.destroyAllWindows()
