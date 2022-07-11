
import cv2
import os
from object import Object
from color import Color
from anchor_manager import AnchorManager
import re


class Tracker:

    FRAME_TITLE = "Object Tracker"
    CAMERA_RES = (640, 480)
    CAMERA_FPS = 30
    RECORD_FORMAT = 'MJPG'
    RECORD_NAME = "bad_example.mp4"
    ESC = 27

    def __init__(self, camera_arg, data_path, is_training, record, show_gui):
        self.is_training = is_training
        self.record = record
        self.show_gui = show_gui

        self.vid_cap = cv2.VideoCapture(camera_arg)
        if not self.vid_cap.isOpened():
            raise Exception("Video file or camera failed to open!")

        self.vid_cap.set(cv2.CAP_PROP_FRAME_WIDTH, Tracker.CAMERA_RES[0])
        self.vid_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Tracker.CAMERA_RES[1])

        self.objects = []
        self.anchor_manager = AnchorManager(self.is_training, camera_arg)

        splits = re.split("\_|\.", camera_arg)

        for color in Color:
            if color == Color.PINK:
                continue
            for split in splits:
                if color.name() == split:
                    self.objects.append(Object(color, self.is_training))
        
        if not os.path.isdir(data_path):
            os.mkdir(data_path)
        
        if self.record:
            fourcc = cv2.VideoWriter_fourcc(*Tracker.RECORD_FORMAT)
            self.out_stream = cv2.VideoWriter(Tracker.RECORD_NAME,
                                            fourcc, Tracker.CAMERA_FPS, Tracker.CAMERA_RES)

    def run(self):
        while True:
            ret, frame = self.vid_cap.read()

            if not ret:
                break

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            self.anchor_manager.reset_ids()
            
            for obj in self.objects:
                found, frame = obj.detect(hsv_frame, frame)
                self.anchor_manager.match(obj, frame, found)

            if self.show_gui:
                cv2.imshow(Tracker.FRAME_TITLE, frame)

            if self.record:
                self.out_stream.write(frame)

            if cv2.waitKey(1) == Tracker.ESC:
                break
        
        self.anchor_manager.save_data()
        if self.record:
            self.out_stream.release()
        self.vid_cap.release()

        return self.anchor_manager.get_f1_score()

    def select_model(self, model_name):
        self.anchor_manager.load_model(model_name)

    def close(self):
        cv2.destroyAllWindows()
