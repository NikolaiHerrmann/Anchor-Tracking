
import cv2
import numpy as np
import os
import sys

sys.path.append(os.path.join("..", "kalman_filter"))
from kf import KF


class Object:

    # Appearance
    RGB_RED = (0, 0, 255)
    RGB_WHITE = (255, 255, 255)
    LINE_THICKNESS = 2
    ALPHA = 0.75
    CIRCLE_RADIUS = 4
    TEXT_SCALE = 1
    UNKNOWN_LABEL = "na"

    # Param
    DIR_BUFFER_SIZE = 30
    BUFFER_CMP_DIS = 10
    DILATE_KERNEL_SIZE = (3, 3)
    DILATE_ITER = 2
    BLUR_KERNEL_SIZE = (7, 7)

    def __init__(self, color, is_training):
        self.color = color
        self.is_training = is_training
        self.id = Object.UNKNOWN_LABEL
        self.hsv_color = color.get_hsv_bounds()
        self.dir_buffer = [(0, 0)] * self.DIR_BUFFER_SIZE
        self.dir_buffer_idx = 0
        self.kf = KF()
        self.kf.predict()

    def _filter_color(self, hsv_frame):
        mask = cv2.inRange(hsv_frame, self.hsv_color[0], self.hsv_color[1])
        mask = cv2.dilate(mask, np.ones(Object.DILATE_KERNEL_SIZE, np.uint8), iterations=Object.DILATE_ITER)
        mask = cv2.GaussianBlur(mask, Object.BLUR_KERNEL_SIZE, 0)
        return mask

    def draw(self, hsv_frame, frame):
        mask = self._filter_color(hsv_frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) < 1:
            return False, frame

        overlay_frame = frame.copy()

        max_contour = max(contours, key=cv2.contourArea)

        # Bounding box
        area_stats = cv2.minAreaRect(max_contour)
        box_points = np.intp(cv2.boxPoints(area_stats))
        cv2.drawContours(overlay_frame, [box_points], 0, Object.RGB_WHITE, thickness=cv2.FILLED)
        overlay_frame = cv2.addWeighted(overlay_frame, Object.ALPHA, frame, 1 - Object.ALPHA, 0)
        cv2.drawContours(overlay_frame, [box_points], 0, Object.RGB_RED, Object.LINE_THICKNESS)

        # ID
        text_coor = min(box_points, key=lambda x: x[1])
        if not self.is_training:
            # text = "tar_id=" + str(self.color.value) + " "
            # if self.id == Object.UNKNOWN_LABEL:
            #     text += Object.UNKNOWN_LABEL
            # else:
            text = "id=" + str(self.id)
            text_color = Object.RGB_WHITE #if self.color == self.id  else Object.RGB_RED
        else:
            text = str(self.color.name)
            text_color = Object.RGB_WHITE
        
        cv2.putText(overlay_frame, text, text_coor, cv2.FONT_HERSHEY_PLAIN, Object.TEXT_SCALE, text_color)

        # Position (center of bounding box)
        cx = np.intp(area_stats[0][0])
        cy = np.intp(area_stats[0][1])
        cv2.circle(overlay_frame, (cx, cy), Object.CIRCLE_RADIUS, Object.RGB_RED, Object.LINE_THICKNESS)

        # Direction Vector
        self.dir_buffer[self.dir_buffer_idx] = (cx, cy)

        dx = self.dir_buffer[self.dir_buffer_idx][0] - self.dir_buffer[self.dir_buffer_idx - Object.BUFFER_CMP_DIS][0]
        dy = self.dir_buffer[self.dir_buffer_idx][1] - self.dir_buffer[self.dir_buffer_idx - Object.BUFFER_CMP_DIS][1]

        if self.dir_buffer_idx == self.DIR_BUFFER_SIZE - 1:  # ensure circular list indexing
            self.dir_buffer_idx = 0
        else:
            self.dir_buffer_idx += 1

        cv2.arrowedLine(overlay_frame, (cx, cy), (cx + dx, cy + dy), Object.RGB_RED, Object.LINE_THICKNESS)

        dir_vec = np.array([dx, dy])

        self.kf.update(cx, cy)
        
        # (x, y) = self.kf.predict()
        # x_ = np.intp(x).item()
        # y_ = np.intp(y).item()
        # cv2.circle(overlay_frame, (x_, y_), 10, (255, 0, 0), 6)

        self.position = np.array([cx, cy])
        self.magnitude = np.linalg.norm(dir_vec)
        self.direction = dir_vec if self.magnitude == 0 else dir_vec / self.magnitude
        self.area = cv2.contourArea(box_points)
        self.rotation = area_stats[2]

        return True, overlay_frame

    def get_thresholds(self, attributes):
        position_thresh = np.linalg.norm(self.position - attributes[0])
        magnitude_thresh = np.abs(self.magnitude - attributes[1])
        direction_thresh = self.direction.dot(attributes[2])
        area_thresh = np.abs(self.area - attributes[3])
        rotation_thresh = np.abs(self.rotation - attributes[4])

        same_color = 1 if self.color == attributes[5] else 0

        return (position_thresh, magnitude_thresh, direction_thresh, area_thresh, rotation_thresh, same_color)

    def get_attributes(self):
        return (self.position, self.magnitude, self.direction, self.area, self.rotation, self.color)
