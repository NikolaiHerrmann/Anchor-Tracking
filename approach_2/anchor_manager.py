
from color import Color
from anchor import Anchor
from detector import Detector
from itertools import combinations
import pandas as pd
import os
import numpy as np
import joblib
import re


class AnchorManager:

    MODEL_PATH = "models"
    MODEL_THRESHOLD = 0.5

    def __init__(self, is_training, file_name):
        self.is_training = is_training
        self.file_name = file_name
        self.tracked_colors = []
        self.anchors = {}
        self.color_switch = {}
        self.data = []
        self.scores = {}

        splits = re.split("\_|\.", file_name)

        for color in Color:
            for split in splits:
                if color.name() == split:
                    self.tracked_colors.append(Detector(color))

    def load_model(self, model_name, path=MODEL_PATH, ext=".pkl"):
        with open(os.path.join(path, model_name + ext), "rb") as f:
            self.model = joblib.load(f)

    def get_objects(self, frame):
        hsv_frame, motion_mask = Detector.get_hsv_and_motion(frame)
        seen_objs = []
        lost_objs = []
        move_objs = []

        for color_obj in self.tracked_colors:
            if color_obj.detect(hsv_frame, motion_mask):
                seen_objs.append(color_obj)
            else:
                lost_objs.append(color_obj)

            if color_obj.should_be_tracked():
                move_objs.append(color_obj)

        return seen_objs, lost_objs, move_objs

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

    # def count_switches(self, obj, id):
    #     val = self.color_switch.get(obj.color)
    #     if val == None:
    #         count = 1
    #         total = 1
    #     else:
    #         prev_id = val[0]
    #         count = val[1]
    #         total = val[2] + 1
    #         if prev_id == id:
    #             count += 1
        
    #     self.color_switch[obj.color] = (id, count, total)

    def match_model(self, obj, t):
        if obj.color == Color.PINK:
            return

        if self.scores.get(obj.color, None) == None:
            self.scores[obj.color] = {"TP": 0, "FN": 0, "FP": 0, "TN": 0, "Stat": 0}
        
        probs = {}

        for idx, anchor in self.anchors.items():
            position_thresh = anchor.get_position_thresh(obj)
            size_thresh = anchor.get_size_thresh(obj)

            features = [position_thresh, size_thresh]
            actual = idx == obj.id

            proba = self.model.predict_proba(np.array([features]))[0]
            # print(position_thresh, size_thresh, proba, obj.color, anchor.track_color)
            if proba[0] < self.MODEL_THRESHOLD:
                probs[idx] = proba[1]
                if actual:
                    self.scores[obj.color]['TP'] += 1
                else:
                    self.scores[obj.color]['FP'] += 1
            else:
                if actual:
                    self.scores[obj.color]['FN'] += 1
                else:
                    self.scores[obj.color]['TN'] += 1
        
        if len(probs) == 0:
            anchor = Anchor(obj, t)
            self.anchors[anchor.get_id()] = anchor
        else:
            idx = max(probs, key=probs.get)
            anchor = self.anchors[idx]
            anchor.update(obj, t)

        id = anchor.get_id()
        obj.set_id(id)

    def match(self, obj, t):
        if self.is_training:
            self.match_train(obj, t)
        else:
            self.match_model(obj, t)

    def track(self, obj, t):
        obj.track()

        if self.is_training:
            anchor_idx = obj.get_track_color()
        else:
            anchor_idx = obj.id
            min_ = 10000000000
            anchor = None

            anchor_ls = sorted(self.anchors.values(), key=lambda x: x.t, reverse=True)
            for a in anchor_ls:
                if a.track_color == obj.color:
                    diff = np.linalg.norm(obj.get_track_position() - a.position)
                    if diff < min_ and diff < 100:
                        min_ = diff
                        anchor = a
                    #print(diff, a.get_id())
            # print("--")
            anchor_true = self.anchors.get(anchor_idx)
            
            if anchor is None:
                return

            anchor.track_update(obj, t)

    def step(self, t, frame):
        seen_objs, _, move_objs = self.get_objects(frame)

        self.check_overlap(seen_objs)

        for color_obj in move_objs:
            self.track(color_obj, t)

        for color_obj in self.tracked_colors:
            color_obj.draw(frame)

            if color_obj.can_be_anchored():
                self.match(color_obj, t)
                #print("anchored ", color_obj.color)

    def save_data(self, dir="data2"):
        idx = self.file_name.find("n/")
        name = self.file_name[idx + 2:idx + 5] + ".csv"

        print("Saving data to: " + name + "\n")

        df = pd.DataFrame(self.data)
        df.columns = ['Position', 'Size', 'Target']

        if not os.path.isdir(dir):
            os.mkdir(dir)

        df.to_csv(os.path.join(dir, name), index=False)

    def recall(self, scores):
        return scores["TP"] / (scores["TP"] + scores["FN"])

    def precision(self, scores):
        return scores["TP"] / (scores["TP"] + scores["FP"])

    def accuracy(self, scores):
        return (scores["TP"] + scores["TN"]) / (scores["TP"] + scores["TN"] + scores["FP"] + scores["FN"])

    def f1_score(self, score):
        recall = self.recall(score)
        precision = self.precision(score)
        return (2 * precision * recall) / (precision + recall)

    def get_f1_score(self):
        scores = self.scores.values()
        scores_len = len(scores)

        recall = 0
        precision = 0
        accuracy = 0
        f1_score = 0

        for color, score in self.scores.items():
            recall += self.recall(score)
            precision += self.precision(score)
            accuracy += self.accuracy(score)
            f1_score += self.f1_score(score)
            print(color, " -> ", score)

        recall /= scores_len
        precision /= scores_len
        accuracy /= scores_len
        f1_score /= scores_len

        return [recall, precision, accuracy, f1_score]
