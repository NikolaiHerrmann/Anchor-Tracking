
import cv2
import numpy as np
import sys
from object import Object
from color import Color
from anchor import Anchor


ESC = 27
SCREEN_MAIN = "objects"


def track(vid_cap):
    #Colors.update_hsv_bounds(Colors.GREEN, np.array([[90, 30, 244], [102, 150, 255]]))
    # objects = []
    # for color in Color:
    #     objects.append(Object(color))

    anchor = Anchor()

    blueObj = Object(Color.BLUE)
    greenObj = Object(Color.GREEN)

    while True:
        _, frame = vid_cap.read()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        found, new_frame = greenObj.draw(hsv_frame, frame)
        found2, new_frame = blueObj.draw(hsv_frame, new_frame)

        if found:
            anchor.match_generate_data(greenObj)

        if found2:
            anchor.match_generate_data(blueObj)

        anchor.data.append("------------")

        cv2.imshow(SCREEN_MAIN, new_frame)
            
        if cv2.waitKey(1) == ESC:
            anchor.save_data()
            break


if __name__ == "__main__":
    camera_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    vid_cap = cv2.VideoCapture(camera_idx)

    track(vid_cap)

    cv2.destroyAllWindows()
    vid_cap.release()