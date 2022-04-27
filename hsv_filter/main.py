
import cv2
import sys
import numpy as np

SCREEN_MAIN = "Filter HSV Bounds"
ESC = 27
EXIT_DELAY = 1

VALUES = {'H min': 179, 'S min': 255, 'V min': 255,
          'H max': 179, 'S max': 255, 'V max': 255}

if __name__ == "__main__":
    camera_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    vid_cap = cv2.VideoCapture(camera_idx)

    cv2.namedWindow(SCREEN_MAIN)

    for key, value in VALUES.items():
        cv2.createTrackbar(key, SCREEN_MAIN, 0, value, lambda x: x)
        if "max" in key:
            cv2.setTrackbarPos(key, SCREEN_MAIN, value)

    values = np.zeros(6, dtype=np.uint8)
    main_values = np.zeros(6, dtype=np.uint8)

    while True:
        _, frame = vid_cap.read()

        for idx, key in enumerate(VALUES.keys()):
            values[idx] = cv2.getTrackbarPos(key, SCREEN_MAIN)

        if not np.array_equal(values, main_values):
            main_values = values.copy()
            print(main_values)

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, values[0:3], values[3:6])
        frame = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow(SCREEN_MAIN, frame)

        if cv2.waitKey(EXIT_DELAY) == ESC:
            break

    cv2.destroyAllWindows()
    vid_cap.release()
