
import cv2
from anchor_manager import AnchorManager


class VideoCapture:

    FRAME_DISPLAY_TIME = 1
    KILL_KEY = 27
    FPS = 30
    RES = (640, 480)

    def __init__(self, input, is_training, show_gui=True, record=False):
        self.cv_capture = cv2.VideoCapture(input)
        if not self.cv_capture.isOpened():
            raise Exception("Video file or camera failed to open!")
        self.is_training = is_training
        self.record = record
        if self.record:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.out_stream = cv2.VideoWriter("vid.mp4", fourcc, 
                                              self.FPS, self.RES)
        self.show_gui = show_gui
        self.anchorManager = AnchorManager(self.is_training, input)

    def read(self):
        t = 0

        while True:
            ret, frame = self.cv_capture.read()
            if not ret:
                break

            self.anchorManager.step(t, frame)

            if self.show_gui:
                cv2.imshow("", frame)
            if self.record:
                self.out_stream.write(frame)

            if cv2.waitKey(VideoCapture.FRAME_DISPLAY_TIME) == VideoCapture.KILL_KEY:
                break

            t += 1

        if self.record:
            self.out_stream.release()

        self.cv_capture.release()

        if self.is_training:
            self.anchorManager.save_data()
            return None
        else:
            return self.anchorManager.get_accuracy()

    def select_model(self, model_name):
        self.anchorManager.load_model(model_name)

    def close():
        cv2.destroyAllWindows()
