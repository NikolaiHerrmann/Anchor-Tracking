import cv2
import sys
import numpy as np

SCREEN_HSV = "hsv"
SCREEN_MASK = "mask"
SCREEN_BLOB = "blob"

ESC = 27
RGB_RED = (0, 0, 255)
RGB_WHITE = (255, 255, 255)
LINE_THICKNESS = 2
DIR_BUFFER_SIZE = 30
ALPHA = 0.75
BUFFER_CMP = 10
CIRCLE_RADIUS = 4


# lower_bound = (78, 104, 114)
# upper_bound = (83, 226, 239)

lower_bound = (90, 30, 244)
upper_bound = (102, 150, 255)


def draw(mask, frame, dir_buffer, dir_buffer_idx):   
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    attributes = None
    overlay_frame = frame.copy()

    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)

        # Bounding box
        box_points = np.intp(cv2.boxPoints(cv2.minAreaRect(max_contour)))
        cv2.drawContours(overlay_frame, [box_points], 0, RGB_WHITE, thickness=cv2.FILLED)
        overlay_frame = cv2.addWeighted(overlay_frame, ALPHA, frame, 1 - ALPHA, 0)
        cv2.drawContours(overlay_frame, [box_points], 0, RGB_RED, LINE_THICKNESS)

        # ID number
        cv2.putText(overlay_frame, "ID: 3", min(box_points, key = lambda x:x[1]), cv2.FONT_HERSHEY_PLAIN, 1.5, RGB_WHITE)

        # 1. Position (center of bounding box)
        moments = cv2.moments(max_contour)
        cx = np.intp(moments['m10'] / moments['m00'])
        cy = np.intp(moments['m01'] / moments['m00'])
        cv2.circle(overlay_frame, (cx, cy), CIRCLE_RADIUS, RGB_RED, LINE_THICKNESS)

        # 2. Direction Vector
        dir_buffer[dir_buffer_idx] = (cx, cy)
        dx = dir_buffer[dir_buffer_idx][0] - dir_buffer[dir_buffer_idx - BUFFER_CMP][0]
        dy = dir_buffer[dir_buffer_idx][1] - dir_buffer[dir_buffer_idx - BUFFER_CMP][1]
        cv2.arrowedLine(overlay_frame, (cx, cy), (cx + dx, cy + dy), RGB_RED, LINE_THICKNESS)

        # 3. Area of bounding box
        size = cv2.contourArea(max_contour)

        attributes = ((cx, cy), (dx, dy), size)
    
    return overlay_frame, attributes


def track(vid_cap):
    kernel = np.ones((3,3), np.uint8)
    dir_buffer = [(0, 0)] * DIR_BUFFER_SIZE
    dir_buffer_idx = 0

    while True:
        ret, frame = vid_cap.read()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.imshow(SCREEN_HSV, hsv)

        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)

        cv2.imshow("mask", mask)


        new_frame, attributes = draw(mask, frame, dir_buffer, dir_buffer_idx)
        cv2.imshow("b", new_frame)

        if (attributes):
            if dir_buffer_idx == DIR_BUFFER_SIZE - 1:
                dir_buffer_idx = 0
            else:
                dir_buffer_idx += 1

        if cv2.waitKey(1) == ESC:
            break


if __name__ == "__main__":
    camera_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    vid_cap = cv2.VideoCapture(camera_idx)

    track(vid_cap)

    cv2.destroyAllWindows()
    vid_cap.release()
