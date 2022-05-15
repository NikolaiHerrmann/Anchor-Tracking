
import sys
import os
import warnings
from tracker import Tracker

DATA_PATH = os.path.join("..", "data")
INPUT_FILE = "track_video_5t.mp4"
IS_TRAINING = False

if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    if IS_TRAINING:
        camera_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    else:
        camera_arg = os.path.join(DATA_PATH, INPUT_FILE)

    tracker = Tracker(camera_arg, DATA_PATH, IS_TRAINING)
    tracker.run()
    tracker.close()