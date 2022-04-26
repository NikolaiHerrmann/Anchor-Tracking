
import cv2
import sys
from object import Object


ESC = 27


def track(vid_cap):

    blueObj = Object((90, 30, 244), (102, 150, 255))
    greenObj = Object((78, 104, 114), (83, 226, 239))

    while True:
        _, frame = vid_cap.read()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        new_frame, _ = blueObj.draw(hsv_frame, frame)
        new_frame, attributes = greenObj.draw(hsv_frame, new_frame)

        cv2.imshow("objects", new_frame)
            
        if cv2.waitKey(1) == ESC:
            break


if __name__ == "__main__":
    camera_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    vid_cap = cv2.VideoCapture(camera_idx)

    track(vid_cap)

    cv2.destroyAllWindows()
    vid_cap.release()