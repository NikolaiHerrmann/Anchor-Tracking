
from color import Color
from anchor import Anchor
from detector import Detector
from itertools import combinations
import pandas as pd
import os
import numpy as np
import joblib


class AnchorManager:

    MODEL_PATH = "models"
    MODEL_THRESHOLD = 0.5

    def __init__(self, is_training, file_name):
        self.is_training = is_training
        self.file_name = file_name
        self.tracked_colors = []
        self.anchors = {}
        self.color_switch = {}
        self.total_switches = 0
        self.data = []

        for color in Color:
            if color.name() in file_name:
                self.tracked_colors.append(Detector(color))

    def load_model(self, model_name, path=MODEL_PATH, ext=".pkl"):
        with open(os.path.join(path, model_name + ext), "rb") as f:
            self.model = joblib.load(f)

    def get_objects(self, frame):
        hsv_frame, motion_mask = Detector.get_hsv_and_motion(frame)
        seen_objs = []
        lost_objs = []

        for color_obj in self.tracked_colors:
            if color_obj.detect(hsv_frame, motion_mask):
                seen_objs.append(color_obj)
            else:
                lost_objs.append(color_obj)

        return seen_objs, lost_objs

    def check_overlap(self, objs):
        comb = list(combinations(objs, 2))

        for obj1, obj2 in comb:
            Detector.overlap(obj1, obj2)

    def match_train(self, obj, t):
        if obj.color == Color.PINK:
            return
        
        for _, anchor in self.anchors.items():

            position_thresh = anchor.get_position_thresh(obj)
            size_thresh = anchor.get_size_thresh(obj)
            # time_thresh = anchor.get_time_thresh(t)

            target = anchor.get_target_class(obj)

            self.data.append([position_thresh, size_thresh, target])

        ground_truth = obj.get_track_color()
        anchor = self.anchors.get(ground_truth)

        if anchor is None:
            anchor = Anchor(obj, t)
        else:
            anchor.update(obj, t)

        self.anchors[ground_truth] = anchor

        obj.set_id(anchor.get_id())

    def count_switches(self, obj, id):
        self.total_switches += 1
        val = self.color_switch.get(obj.color)
        if val == None:
            self.color_switch[obj.color] = (id, 0)
        else:
            prev_id = val[0]
            count = val[1] 
            if prev_id == id:
                count += 1
            self.color_switch[obj.color] = (id, count)

    def match_model(self, obj, t):
        probs = {}

        for idx, anchor in self.anchors.items():
            position_thresh = anchor.get_position_thresh(obj)
            size_thresh = anchor.get_size_thresh(obj)

            features = [position_thresh, size_thresh]

            proba = self.model.predict_proba(np.array([features]))[0]
            # print(position_thresh, size_thresh, proba, obj.color, anchor.track_color)
            if proba[0] < self.MODEL_THRESHOLD:
                probs[idx] = proba[1]
        
        if len(probs) == 0:
            anchor = Anchor(obj, t)
            self.anchors[anchor.get_id()] = anchor
        else:
            idx = max(probs, key=probs.get)
            anchor = self.anchors[idx]
            anchor.update(obj, t)

        id = anchor.get_id()
        obj.set_id(id)
        self.count_switches(obj, id)

    def match(self, obj, t):
        if self.is_training:
            self.match_train(obj, t)
        else:
            self.match_model(obj, t)

    def track_train(self, obj, t):
        obj.track()
        if self.is_training:
            anchor_idx = obj.get_track_color()
        else:
            anchor_idx = obj.id
        anchor = self.anchors.get(anchor_idx)

        if anchor is None:
            return

        anchor.track_update(obj, t)

    def step(self, t, frame):
        seen_objs, lost_objs = self.get_objects(frame)

        self.check_overlap(seen_objs)

        for color_obj in self.tracked_colors:            
            self.track_train(color_obj, t)
            color_obj.draw(frame)

            if color_obj.can_be_anchored():
                self.match(color_obj, t)

    def save_data(self, dir="data"):
        idx = self.file_name.find("tv")
        name = self.file_name[idx: idx + 3] + ".csv"

        print("Saving data to: " + name + "\n")

        df = pd.DataFrame(self.data)
        df.columns = ['Position', 'Size', 'Target']

        if not os.path.isdir(dir):
            os.mkdir(dir)

        df.to_csv(os.path.join(dir, name), index=False)

    def get_accuracy(self):
        total_correct_class = 0

        for color in self.color_switch.values():
            total_correct_class += color[1]

        accuracy = total_correct_class / self.total_switches
        return accuracy
