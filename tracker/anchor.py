
from scipy.fft import idstn
from color import Color
import numpy as np
import pandas as pd
import joblib
import os


class Anchor:

    def __init__(self, is_training):
        self.is_training = is_training
        self.anchors = {}
        if not self.is_training:
            self.load_model("knn")
            self.correct_pred = 0
            self.total_pred = 0
            self.maxId = 0
        else:
            self.data = []

    def load_model(self, model_name, dir="model", ext=".pkl"):
        path = os.path.join("..", dir)
        with open(os.path.join(path, model_name + ext), "rb") as f:
            self.model = joblib.load(f)

    def training_match(self, obj):
        for anchor in self.anchors.keys():

            anchor_attributes = self.anchors[anchor]
            thresholds = obj.get_thresholds(anchor_attributes)

            self.data.append(thresholds)

        self.anchors[obj.color.value] = obj.get_attributes()

    def model_match(self, obj):
        probs = {}
        
        for anchor in self.anchors.keys():
            anchor_attributes = self.anchors[anchor]

            thresholds = obj.get_thresholds(anchor_attributes)

            features = np.array([thresholds[0:5]])
            target = thresholds[5]

            pred = self.model.predict(features)[0]
            if pred == 1:
                probs[anchor] = self.model.predict_proba(features)[0][1]

            self.correct_pred += 1 if pred == target else 0
            self.total_pred += 1
        
        new_attributes = obj.get_attributes()
        if len(probs) == 0: # acquire
            self.anchors[self.maxId] = new_attributes
            self.maxId += 1
            id = self.maxId
        else: # require
            id = max(probs, key=probs.get)
            self.anchors[id] = new_attributes

        obj.id = id

    def match(self, obj):
        if self.is_training:
            self.training_match(obj)
        else:
            self.model_match(obj)

    def get_accuracy(self):
        if self.is_training:
            raise Exception("No accuracy while training!")
        if self.total_pred == 0:
            return 0
        return self.correct_pred / self.total_pred 

    def save_data(self, name="tracking_data", dir="data"):
        if not self.is_training or len(self.data) == 0:
            print("No training data was saved!")
            return

        df = pd.DataFrame(self.data)
        df.columns = ['position_thresh', 'magnitude_thresh', 'direction_thresh', 'area_thresh', 'rotation_thresh', 'class']

        path = os.path.join("..", dir)
        if not os.path.isdir(path):
            os.mkdir(path)

        df.to_csv(os.path.join(path, name + ".csv"), index=False)
