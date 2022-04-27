
import cv2
import sys
from tracker import Tracker

if __name__ == "__main__":
    camera_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    vid_cap = cv2.VideoCapture(camera_idx)

    tracker = Tracker(vid_cap)
    tracker.run()

    cv2.destroyAllWindows()
    vid_cap.release()