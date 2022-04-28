
import sys
import warnings
from tracker import Tracker

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    camera_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    tracker = Tracker(camera_idx)
    tracker.run()