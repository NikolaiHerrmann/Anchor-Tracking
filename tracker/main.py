
import sys
import os
import warnings
from tracker import Tracker

IS_TRAINING = False

if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    if IS_TRAINING:
        camera_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    else:
        camera_arg = os.path.join("..", Tracker.DATA_PATH, Tracker.RECORD_NAME)

    tracker = Tracker(camera_arg, IS_TRAINING)
    tracker.run()