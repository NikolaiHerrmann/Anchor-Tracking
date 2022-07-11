"""
Adapted from https://stackoverflow.com/questions/57469394/opencv-choosing-hsv-thresholds-for-color-filtering
         and https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html
"""

import cv2
import sys
import numpy as np

SCREEN_MAIN = "Filter HSV Bounds"
ESC = 27
TEXT_COL = (255, 255, 255)
TEXT_LOC = (100, 50)
TEXT_DIM = 2

VALUES = {'H min': 179, 'S min': 255, 'V min': 255,
          'H max': 179, 'S max': 255, 'V max': 255}

if __name__ == "__main__":
    camera_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    vid_cap = cv2.VideoCapture("test.mp4")

    cv2.namedWindow(SCREEN_MAIN)

    for key, value in VALUES.items():
        init_val = value if "max" in key else 0
        cv2.createTrackbar(key, SCREEN_MAIN, init_val, value, lambda x: x)

    values = np.zeros(6, dtype=np.uint8)
    main_values = np.zeros(6, dtype=np.uint8)

    count = 0

    while True:
        if count <= 0:  # only look at first frame (remove to play video)
            ret, frame_ = vid_cap.read()

        frame = frame_.copy()
        count += 1

        for idx, key in enumerate(VALUES.keys()):
            values[idx] = cv2.getTrackbarPos(key, SCREEN_MAIN)

        if not np.array_equal(values, main_values):
            main_values = values.copy()
            print(main_values)

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, values[0:3], values[3:6])

        frame = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.putText(frame, str(main_values), TEXT_LOC,
                    cv2.FONT_HERSHEY_PLAIN, TEXT_DIM, TEXT_COL)

        cv2.imshow(SCREEN_MAIN, frame)

        if cv2.waitKey(1) == ESC:
            break

    vid_cap.release()
    cv2.destroyAllWindows()
