
from color import Color
import numpy as np
import pandas as pd
import joblib
import os


class Anchor:

    def __init__(self, is_training=False):
        self.is_training = is_training
        self.anchors = [None] * len(Color)
        if not self.is_training:
            self.load_model("knn")
            self.correct_pred = 0
            self.total_pred = 0
        else:
            self.data = []

    def load_model(self, model_name, dir="model", ext=".pkl"):
        path = os.path.join("..", dir)
        with open(os.path.join(path, model_name + ext), "rb") as f:
            self.model = joblib.load(f)

    def match(self, obj):
        for anchor_attributes in self.anchors:
            if not anchor_attributes:
                continue

            thresholds = obj.get_thresholds(anchor_attributes)

            if self.is_training:
                self.data.append(thresholds)
            else:
                target = thresholds[5]
                features = np.array([thresholds[0:5]])

                pred = self.model.predict(features)

                self.correct_pred += 1 if pred == target else 0
                self.total_pred += 1

                if pred == 1:
                    obj.id = anchor_attributes[5]
                    break

        self.anchors[obj.color.value] = obj.get_attributes()

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
