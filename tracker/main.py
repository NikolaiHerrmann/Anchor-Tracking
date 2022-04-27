
import sys
from tracker import Tracker

if __name__ == "__main__":
    camera_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    tracker = Tracker(camera_idx)
    tracker.run()