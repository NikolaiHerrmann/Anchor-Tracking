
import cv2
import numpy as np
from kalman import KF
from color import Color

class Detector:

    MORPH_KERNEL_SIZE = (3, 3)
    MORPH_ITER = 3
    MOTION_MORPH_ITER = 5
    BACK_SUBTRACTOR = cv2.createBackgroundSubtractorKNN(history=20)
    MOTION_THRESHOLD = 100
    DIM = (480, 640)

    def __init__(self, color):
        self.color = color
        self.hsv_bounds = color.hsv()
        self.found_color = False

        self.possible_contour = None
        self.possible_contour_mask = None

        self.contour = None
        self.contour_mask = None

        self.in_motion = 0
        self.is_overlapping = 0
        self.is_resting = 0
        self.is_anchorable = False
        self.track_position = np.array([0, 0])

        self.x = self.DIM[1] + 1
        self.y = self.x

        self.id = -1

        self.kf = KF()

    def detect(self, hsv_frame, motion_mask):
        hsv_mask = self.get_hsv_mask(hsv_frame)        
        self.found_color = cv2.countNonZero(hsv_mask) != 0

        if not self.found_color:
            return False

        contours, _ = cv2.findContours(hsv_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)

        ## needed for when all 3 cards are blue
        # contours = sorted(contours, key=lambda x: cv2.contourArea(x))

        # if np.count_nonzero(self.track_position) != 0:
        #     found = False
        #     for cnt in contours:
        #         pos = Detector.get_position_contour(cnt)
        #         diff = np.linalg.norm(pos - self.track_position)
        #         area = cv2.contourArea(cnt)
        #         if diff < 100 and area > 6000:
        #             self.possible_contour = cnt
        #             #print("found: ", diff)
        #             found = True
        #             break
        #         #print(diff)
        #     if not found:
        #         if self.color == Color.PINK:
        #             self.possible_contour = max(contours, key=cv2.contourArea)
        #         else:
        #             self.found_color = False
        #         return False
        # else:           
        #     self.possible_contour = max(contours, key=cv2.contourArea)


        self.possible_contour = max(contours, key=cv2.contourArea)
        self.possible_area = cv2.contourArea(self.possible_contour)
        if self.possible_area < 1000:
            self.found_color = False
            return False
        self.possible_contour_mask = cv2.fillPoly(np.zeros(self.DIM), 
                                                  [self.possible_contour], 1)
        
        cv2.fillPoly(hsv_frame, [self.possible_contour], color=(0, 0, 0))
        # cv2.imshow("c", hsv_frame)

        and_mask = np.logical_and(self.possible_contour_mask, motion_mask)
        future_in_motion = 1 if np.count_nonzero(and_mask == True) > \
                                self.MOTION_THRESHOLD else 0



        self.is_anchorable = False
        
        if future_in_motion:
            self.is_resting = 0

        if (self.in_motion and not future_in_motion) or self.contour is None:
            self.contour = self.possible_contour
            self.contour_mask = self.possible_contour_mask
            self.area = self.possible_area
            self.is_anchorable = True

        self.in_motion = future_in_motion
        self.is_overlapping = 0

        return True

    def draw(self, draw_frame):
        if self.contour is not None:

            if not self.in_motion and not self.is_overlapping and not self.is_resting:
                self.contour = self.possible_contour
                self.contour_mask = self.possible_contour_mask
                self.area = self.possible_area
                self.is_resting = 1
                self.is_anchorable = True

            if not self.in_motion:
                self.draw_bounding_box(draw_frame)
            else:
                self.draw_prediction(draw_frame)

    def can_be_anchored(self):
        return self.is_anchorable

    def should_be_tracked(self):
        return self.in_motion

    def track(self):
        self.kf.update(self.track_position[0], self.track_position[1])

        if self.found_color:
            if self.in_motion:
                contour = self.possible_contour
                self.track_position = self.get_position_contour(contour)                
        else:
            self.track_position = np.array(self.kf.predict())
            # print("no track position ", self.track_position)
            return

    @staticmethod
    def overlap(obj1, obj2):
        obj1_mask = obj1.possible_contour_mask if obj1.in_motion else obj1.contour_mask
        obj2_mask = obj2.possible_contour_mask if obj2.in_motion else obj2.contour_mask

        overlap_mask = np.logical_and(obj1_mask, obj2_mask)
        is_overlap = 1 if (overlap_mask).any() else 0

        obj1.is_overlapping |= is_overlap
        obj2.is_overlapping |= is_overlap

    def draw_prediction(self, draw_frame, radius_size=8):
        #print(np.intp(self.track_position[0]), np.intp(self.track_position[1]))
        cx = np.intp(self.track_position[0])
        cy = np.intp(self.track_position[1])
        if self.color != Color.PINK:
            if abs(cx) < 600 and abs(cy) < 600:
                cv2.circle(draw_frame, (cx, cy), 
                        radius_size, self.color.bgr(), 2)

    def draw_bounding_box(self, draw_frame, rect_offset=40, add_= 8):
        if self.color != Color.PINK:
            contour = self.contour
            color = self.color.bgr()
            x, y, width, height = cv2.boundingRect(contour)
            cv2.rectangle(draw_frame, (x - rect_offset, y - rect_offset),
                        (x + width + rect_offset, y + height + rect_offset), color, 2)
            cv2.drawContours(draw_frame, [contour], -1, color, 2)
            cv2.putText(draw_frame, "id " + str(self.id), (x - 10, y + height + 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2)

    @staticmethod
    def get_position_contour(contour):
        moments = cv2.moments(contour)
        x = moments["m10"] / moments["m00"]
        y = moments["m01"] / moments["m00"]
        return np.array([x, y])

    def get_position(self):
        return self.get_position_contour(self.contour)

    def get_track_position(self):
        return self.track_position

    def get_size(self):
        return self.area

    def set_id(self, id):
        self.id = id

    def get_track_color(self):
        return self.color

    def get_hsv_mask(self, hsv_frame):
        mask = cv2.inRange(hsv_frame, self.hsv_bounds[0], self.hsv_bounds[1])
        mask = cv2.dilate(mask, np.ones((self.MORPH_KERNEL_SIZE), np.uint8),
                          iterations=self.MORPH_ITER)
        return mask

    @staticmethod
    def get_motion_mask(frame):
        motion_mask = Detector.BACK_SUBTRACTOR.apply(frame)
        motion_mask = cv2.erode(motion_mask, np.ones((Detector.MORPH_KERNEL_SIZE), np.uint8),
                                iterations=Detector.MOTION_MORPH_ITER)
        motion_mask[motion_mask == 127] = 0  # remove shadows
        return motion_mask

    @staticmethod
    def get_hsv_and_motion(frame):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        motion_mask = Detector.get_motion_mask(frame)
        return hsv_frame, motion_mask
