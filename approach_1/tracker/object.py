
import cv2
import numpy as np


class Object:

    # Param
    DIR_BUFFER_SIZE = 30
    BUFFER_CMP_DIS = 10
    DILATE_KERNEL_SIZE = (3, 3)
    DILATE_ITER = 2
    BLUR_KERNEL_SIZE = (7, 7)

    # Appearance
    RGB_RED = (0, 0, 255)
    RGB_WHITE = (255, 255, 255)
    RGB_BLUE = (255, 0, 0)
    RGB_GREEN = (0, 255, 0)
    LINE_THICKNESS = 2
    ALPHA = 0.75
    CIRCLE_RADIUS = 4
    TEXT_SCALE = 0.6

    def __init__(self, color, is_training):
        self.color = color
        self.is_training = is_training
        self.hsv_color = color.get_hsv_bounds()
        self.dir_buffer = [(0, 0)] * self.DIR_BUFFER_SIZE
        self.dir_buffer_idx = 0
        self.prev_id = -1

    def _filter_color(self, hsv_frame):
        mask = cv2.inRange(hsv_frame, self.hsv_color[0], self.hsv_color[1])
        mask = cv2.dilate(mask, np.ones(Object.DILATE_KERNEL_SIZE, np.uint8),
                          iterations=Object.DILATE_ITER)
        mask = cv2.GaussianBlur(mask, Object.BLUR_KERNEL_SIZE, 0)
        return mask

    def draw_kalman_prediction(self, frame, x, y):
        x_ = np.intp(x)
        y_ = np.intp(y)
        cv2.circle(frame, (x_, y_), Object.CIRCLE_RADIUS * 2,
                   Object.RGB_BLUE, Object.LINE_THICKNESS * 2)

    def draw_id(self, id, frame):
        # find min y value
        text_coor = min(self.box_points, key=lambda x: x[1])
        text = str(id)
        # cv2.putText(frame, text, text_coor, cv2.FONT_HERSHEY_SIMPLEX,
        #             Object.TEXT_SCALE, Object.RGB_GREEN, Object.LINE_THICKNESS)

    def detect(self, hsv_frame, frame):
        mask = self._filter_color(hsv_frame)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) < 1:
            return False, frame

        overlay_frame = frame.copy()

        max_contour = max(contours, key=cv2.contourArea) # = cv2.findNonZero(mask)

        # Bounding box
        area_stats = cv2.minAreaRect(max_contour)
        self.box_points = np.intp(cv2.boxPoints(area_stats))
        # cv2.drawContours(overlay_frame, [self.box_points], 0, 
        #                  Object.RGB_WHITE, thickness=cv2.FILLED)
        # overlay_frame = cv2.addWeighted(overlay_frame, Object.ALPHA, frame, 
        #                                 1 - Object.ALPHA, 0)
        # cv2.drawContours(overlay_frame, [self.box_points], 0, 
        #                  Object.RGB_RED, Object.LINE_THICKNESS)

        # Position (center of bounding box)
        cx = np.intp(area_stats[0][0])
        cy = np.intp(area_stats[0][1])
        # cv2.circle(overlay_frame, (cx, cy), Object.CIRCLE_RADIUS,
        #            Object.RGB_RED, Object.LINE_THICKNESS)

        contour = max_contour
        #color = self.color.bgr()
        color = (0, 0, 255)
        rect_offset = 40
        x, y, width, height = cv2.boundingRect(contour)
        cv2.rectangle(overlay_frame, (x - rect_offset, y - rect_offset),
                      (x + width + rect_offset, y + height + rect_offset), color, 2)
        cv2.drawContours(overlay_frame, [contour], -1, color, 2)
        cv2.putText(overlay_frame, "id " + str(self.prev_id), (x - 10, y + height + 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)

        # Direction Vector
        self.dir_buffer[self.dir_buffer_idx] = (cx, cy)

        dx = self.dir_buffer[self.dir_buffer_idx][0] - self.dir_buffer[self.dir_buffer_idx - Object.BUFFER_CMP_DIS][0]
        dy = self.dir_buffer[self.dir_buffer_idx][1] - self.dir_buffer[self.dir_buffer_idx - Object.BUFFER_CMP_DIS][1]

        if self.dir_buffer_idx == self.DIR_BUFFER_SIZE - 1:  # ensure circular list indexing
            self.dir_buffer_idx = 0
        else:
            self.dir_buffer_idx += 1

        cv2.arrowedLine(overlay_frame, (cx, cy), (cx + dx, cy + dy),
                        Object.RGB_RED, Object.LINE_THICKNESS)

        dir_vec = np.array([dx, dy])

        self.position = np.array([cx, cy])
        self.magnitude = np.linalg.norm(dir_vec)
        self.direction = dir_vec if self.magnitude == 0 else dir_vec / self.magnitude
        self.area = cv2.contourArea(self.box_points)
        self.rotation = area_stats[2]

        return True, overlay_frame

    def get_thresholds(self, attributes):
        position_thresh = np.linalg.norm(self.position - attributes[0])
        magnitude_thresh = np.abs(self.magnitude - attributes[1])
        direction_thresh = self.direction.dot(attributes[2])
        area_thresh = np.abs(self.area - attributes[3])
        rotation_thresh = np.abs(self.rotation - attributes[4])

        class_ = 1 if self.color == attributes[5] else 0

        return [position_thresh, magnitude_thresh, direction_thresh, area_thresh, rotation_thresh, class_]

    def get_attributes(self):
        return [self.position, self.magnitude, self.direction, self.area, self.rotation, self.color]
