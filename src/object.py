
import cv2
import numpy as np

class Object:

    # Appearance
    RGB_RED = (0, 0, 255)
    RGB_WHITE = (255, 255, 255)
    LINE_THICKNESS = 2
    ALPHA = 0.75
    CIRCLE_RADIUS = 4

    # Settings
    DIR_BUFFER_SIZE = 30
    _ID = 1

    # Param
    BUFFER_CMP = 10
    DILATE_KERNEL = np.ones((3, 3), np.uint8)
    BLUR_KERNEL_SIZE = (7, 7)

    def __init__(self, hsv_low, hsv_high):
        self.hsv_low = hsv_low
        self.hsv_high = hsv_high
        self.debug = False
        self.dir_buffer = [(0, 0)] * self.DIR_BUFFER_SIZE
        self.dir_buffer_idx = 0
        self.id = Object._ID
        Object._ID += 1

    def _filter_color(self, hsv_frame):
        mask = cv2.inRange(hsv_frame, self.hsv_low, self.hsv_high)
        mask = cv2.dilate(mask, Object.DILATE_KERNEL, iterations = 2)
        mask = cv2.GaussianBlur(mask, Object.BLUR_KERNEL_SIZE, 0)

        if self.debug:
            cv2.imshow("Mask " + str(self.id), mask)
        
        return mask

    def draw(self, hsv_frame, frame):  
        mask = self._filter_color(hsv_frame) 
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        attributes = None
        overlay_frame = frame.copy()

        if len(contours) > 0:
            max_contour = max(contours, key = cv2.contourArea)

            # Bounding box
            box_points = np.intp(cv2.boxPoints(cv2.minAreaRect(max_contour)))
            cv2.drawContours(overlay_frame, [box_points], 0, Object.RGB_WHITE, thickness=cv2.FILLED)
            overlay_frame = cv2.addWeighted(overlay_frame, Object.ALPHA, frame, 1 - Object.ALPHA, 0)
            cv2.drawContours(overlay_frame, [box_points], 0, Object.RGB_RED, Object.LINE_THICKNESS)

            # ID number
            cv2.putText(overlay_frame, "ID: " + str(self.id), min(box_points, key = lambda x:x[1]), cv2.FONT_HERSHEY_PLAIN, 1.5, Object.RGB_WHITE)

            # 1. Position (center of bounding box)
            moments = cv2.moments(max_contour)
            cx = np.intp(moments['m10'] / moments['m00'])
            cy = np.intp(moments['m01'] / moments['m00'])
            cv2.circle(overlay_frame, (cx, cy), Object.CIRCLE_RADIUS, Object.RGB_RED, Object.LINE_THICKNESS)

            # 2. Direction Vector
            self.dir_buffer[self.dir_buffer_idx] = (cx, cy)
            
            dx = self.dir_buffer[self.dir_buffer_idx][0] - self.dir_buffer[self.dir_buffer_idx - self.BUFFER_CMP][0]
            dy = self.dir_buffer[self.dir_buffer_idx][1] - self.dir_buffer[self.dir_buffer_idx - self.BUFFER_CMP][1]
            
            if self.dir_buffer_idx == self.DIR_BUFFER_SIZE - 1:
                self.dir_buffer_idx = 0
            else:
                self.dir_buffer_idx += 1
            
            cv2.arrowedLine(overlay_frame, (cx, cy), (cx + dx, cy + dy), Object.RGB_RED, Object.LINE_THICKNESS)

            # 3. Area of bounding box
            size = cv2.contourArea(max_contour)

            attributes = (np.array([cx, cy]), Object.normalize(np.array([dx, dy])), size)
        
        return overlay_frame, attributes

    def normalize(vec):
        norm = np.linalg.norm(vec)
        return vec if norm == 0 else vec / norm