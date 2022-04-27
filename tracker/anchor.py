
from color import Color
import numpy as np
import pandas as pd
import joblib
import os
import warnings


class Anchor:

    def __init__(self):
        self.anchors = [None] * len(Color)
        self.data = []
        self.load_model("knn")
        warnings.filterwarnings('ignore')

    def load_model(self, model_name, dir="model", ext=".pkl"):
        path = os.path.join("..", dir)
        with open(os.path.join(path, model_name + ext), "rb") as f:
            self.model = joblib.load(f)

    def match_generate_data(self, obj):
        for anchor_attributes in self.anchors:
            if not anchor_attributes:
                continue

            thresholds = obj.get_thresholds(anchor_attributes)

            true = thresholds[5]
            subset = np.array([thresholds[0:5]])

            pred = self.model.predict(subset)
            print(true == pred)

            self.data.append(thresholds)

            # possibly remove anchor if old

        self.anchors[obj.color.value] = obj.get_attributes()

    def save_data(self, name="tracking_data", dir="data"):
        if len(self.data) == 0:
            return
        df = pd.DataFrame(self.data)
        df.columns = ['position_thresh', 'magnitude_thresh', 'direction_thresh', 'area_thresh', 'rotation_thresh', 'class']
        path = os.path.join("..", dir)
        if not os.path.isdir(path):
            os.mkdir(path)
        df.to_csv(os.path.join(path, name + ".csv"), index=False)
