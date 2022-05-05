
import numpy as np
import cv2
from kf import KF


class Object:

    DILATE_KERNEL_SIZE = (3, 3)
    DILATE_ITER = 2
    BLUR_KERNEL_SIZE = (7, 7)

    def __init__(self):
        self.hsv_color = np.array([[0, 137, 173], [179, 255, 255]])

    def _filter_color(self, hsv_frame):
        mask = cv2.inRange(hsv_frame, self.hsv_color[0], self.hsv_color[1])
        mask = cv2.dilate(mask, np.ones(Object.DILATE_KERNEL_SIZE, np.uint8), iterations=Object.DILATE_ITER)
        mask = cv2.GaussianBlur(mask, Object.BLUR_KERNEL_SIZE, 0)
        return mask

    def find(self, hsv_frame, frame):
        mask = self._filter_color(hsv_frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) < 1:
            return False, None, None

        max_contour = max(contours, key=cv2.contourArea)

        area_stats = cv2.minAreaRect(max_contour)
        box_points = np.intp(cv2.boxPoints(area_stats))
        cv2.drawContours(frame, [box_points], 0, (255, 0, 0), 1)

        cx = area_stats[0][0]
        cy = area_stats[0][1]

        return True, np.array([[cx], [cy]]), (np.intp(cx), np.intp(cy))

def run():
    vid_cap = cv2.VideoCapture(2)
    kf = KF()
    obj = Object()

    while True:

        _, frame = vid_cap.read()

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        found, center, c = obj.find(hsv_frame, frame)

        (x, y) = kf.predict()
        cx = np.intp(x).item()
        cy = np.intp(y).item()
        cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 6)
        
        if found:
            cv2.circle(frame, c, 10, (255, 0, 0), 1)
            kf.update(center)

        cv2.imshow("", frame)

        if cv2.waitKey(1) == 27:
            break

if __name__ == "__main__":
    run()


