
import numpy as np
import pandas as pd
import joblib
import os


class AnchorManager:

    def __init__(self, is_training, file_name):
        self.is_training = is_training
        self.file_name = file_name
        self.anchors = {}
        self.ids_taken = set()
        self.scores = {}

        if not self.is_training:
            self.load_model("Logistic_Regression")
            self.maxId = 1
        else:
            self.data = []

    def load_model(self, model_name, dir="models", ext=".pkl"):
        path = os.path.join("..", dir)
        with open(os.path.join(path, model_name + ext), "rb") as f:
            self.model = joblib.load(f)

    def training_match(self, obj, found):
        if not found:
            return
        
        for anchor in self.anchors.keys():

            anchor_attributes = self.anchors[anchor]
            thresholds = obj.get_thresholds(anchor_attributes)

            self.data.append(thresholds)

        self.anchors[obj.color.value] = obj.get_attributes()

    def model_match(self, obj, frame, found):
        if not found:
            return
        
        probs = {}

        if self.scores.get(obj.color, None) == None:
            self.scores[obj.color] = {"TP": 0, "FN": 0, "FP": 0, "TN": 0, "Stat": 0}
        
        # if obj.magnitude <= 5:
        #     self.scores[obj.color]["Stat"] += 1
        #     flag = True
        # else:
        flag = False

        for anchor_idx, anchor_attributes in self.anchors.items():

            if anchor_idx in self.ids_taken:
                continue
                
            thresholds = obj.get_thresholds(anchor_attributes)
            actual = obj.prev_id == anchor_idx
            features = np.array([thresholds[0:5]])
            
            proba = self.model.predict_proba(features)[0]

            if proba[0] <= 0.5:
                probs[anchor_idx] = proba[1]
                if not flag:
                    if actual:
                        self.scores[obj.color]['TP'] += 1
                    else:
                        self.scores[obj.color]['FP'] += 1
            else:
                if not flag:
                    if actual:
                        self.scores[obj.color]['FN'] += 1
                    else:
                        self.scores[obj.color]['TN'] += 1
   
        new_attributes = obj.get_attributes()

        if len(probs) == 0: # acquire
            self.anchors[self.maxId] = new_attributes
            self.maxId += 1
            id = self.maxId
        else:               # re-acquire
            id = max(probs, key=probs.get)
            self.anchors[id] = new_attributes

        self.ids_taken.add(id)
        obj.draw_id(id, frame)
        obj.prev_id = id

    def reset_ids(self):
        self.ids_taken.clear()

    def match(self, obj, frame, found):
        if self.is_training:
            self.training_match(obj, found)
        else:
            self.model_match(obj, frame, found)

    def save_data(self, dir="data"):
        if not self.is_training or len(self.data) == 0:
            # print("No training data could be saved!")
            return

        idx = self.file_name.find("n/")
        name = self.file_name[idx + 2:idx + 5] + ".csv"

        print("Saving data to: " + name + "\n")

        df = pd.DataFrame(self.data)
        df.columns = ['Position', 'Magnitude', 'Direction', 'Area', 'Rotation', 'class']

        path = os.path.join("..", dir)
        if not os.path.isdir(path):
            os.mkdir(path)

        df.to_csv(os.path.join(path, name + ".csv"), index=False)

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
