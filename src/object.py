
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
    BUFFER_CMP_DIS = 10
    DILATE_KERNEL = np.ones((3, 3), np.uint8)
    BLUR_KERNEL_SIZE = (7, 7)

    def __init__(self, track_color):
        self.hsv_color = track_color.get_hsv_bounds()
        self.debug = False
        self.dir_buffer = [(0, 0)] * self.DIR_BUFFER_SIZE
        self.dir_buffer_idx = 0

        self.id = Object._ID
        Object._ID += 1

    def _filter_color(self, hsv_frame):
        mask = cv2.inRange(hsv_frame, self.hsv_color[0], self.hsv_color[1])
        mask = cv2.dilate(mask, Object.DILATE_KERNEL, iterations = 2)
        mask = cv2.GaussianBlur(mask, Object.BLUR_KERNEL_SIZE, 0)

        if self.debug:
            cv2.imshow("Mask " + str(self.id), mask)
        
        return mask

    def draw(self, hsv_frame, frame):  
        mask = self._filter_color(hsv_frame) 
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        overlay_frame = frame.copy()

        if len(contours) > 0:
            max_contour = max(contours, key = cv2.contourArea)

            # Bounding box
            area_stats = cv2.minAreaRect(max_contour)
            box_points = np.intp(cv2.boxPoints(area_stats))
            cv2.drawContours(overlay_frame, [box_points], 0, Object.RGB_WHITE, thickness = cv2.FILLED)
            overlay_frame = cv2.addWeighted(overlay_frame, Object.ALPHA, frame, 1 - Object.ALPHA, 0)
            cv2.drawContours(overlay_frame, [box_points], 0, Object.RGB_RED, Object.LINE_THICKNESS)

            # ID number
            text_coor = min(box_points, key = lambda x : x[1])
            cv2.putText(overlay_frame, "ID: " + str(self.id), text_coor, cv2.FONT_HERSHEY_PLAIN, 1.5, Object.RGB_WHITE)

            # Position (center of bounding box)
            cx = np.intp(area_stats[0][0])
            cy = np.intp(area_stats[0][1])
            cv2.circle(overlay_frame, (cx, cy), Object.CIRCLE_RADIUS, Object.RGB_RED, Object.LINE_THICKNESS)

            # Direction Vector
            self.dir_buffer[self.dir_buffer_idx] = (cx, cy)
            
            dx = self.dir_buffer[self.dir_buffer_idx][0] - self.dir_buffer[self.dir_buffer_idx - Object.BUFFER_CMP_DIS][0]
            dy = self.dir_buffer[self.dir_buffer_idx][1] - self.dir_buffer[self.dir_buffer_idx - Object.BUFFER_CMP_DIS][1]
            
            if self.dir_buffer_idx == self.DIR_BUFFER_SIZE - 1: # ensure circular list indexing
                self.dir_buffer_idx = 0
            else:
                self.dir_buffer_idx += 1
            
            cv2.arrowedLine(overlay_frame, (cx, cy), (cx + dx, cy + dy), Object.RGB_RED, Object.LINE_THICKNESS)

            dir_vec = np.array([dx, dy])

            self.position = np.array([cx, cy])
            self.magnitude = np.linalg.norm(dir_vec)
            self.direction = dir_vec if self.magnitude == 0 else dir_vec / self.magnitude
            self.area = cv2.contourArea(box_points)
            self.rotation = area_stats[2]

            print(self.position, self.magnitude, self.direction, self.area, self.rotation)

        return overlay_frame

    def compare(self, other):
        position_thresh = np.linalg.norm(self.position - other.position)
        magnitude_thresh = np.abs(self.magnitude - other.magnitude)
        direction_thresh = self.direction.dot(other.direction)
        area_thresh = np.abs(self.area - other.area)
        rotation_thresh = np.abs(self.rotation - other.rotation)
        
        return (position_thresh, magnitude_thresh, direction_thresh, area_thresh, rotation_thresh)
