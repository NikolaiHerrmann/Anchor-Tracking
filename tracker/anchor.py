
import numpy as np
import pandas as pd
import joblib
import os
import cv2


class Anchor:

    def __init__(self, is_training):
        self.is_training = is_training
        self.anchors = {}
        if not self.is_training:
            self.load_model("logistic_regression")
            self.correct_pred = 0
            self.total_pred = 0
            self.maxId = 1
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

    def model_match(self, obj, frame):
        probs = {}

        (x, y) = obj.kf.predict()
        x_ = np.intp(x).item()
        y_ = np.intp(y).item()
        cv2.circle(frame, (x_, y_), 10, (255, 0, 0), 6)
        
        for anchor in self.anchors.keys():
            anchor_attributes = self.anchors[anchor]

            thresholds = obj.get_thresholds(anchor_attributes)

            features = np.array([thresholds[0:5]])
            # target = thresholds[5]
            
            pred = self.model.predict(features)[0]

            if pred == 1:
                probs[anchor] = self.model.predict_proba(features)[0][1]
                #self.total_pred += 1
            # else: # will always go in with multiple objects
            #     diff = np.linalg.norm(np.array([x_, y_]) - anchor_attributes[0])
                
            #     print(diff)
            #     print(obj.color, anchor_attributes[5])
            #     if diff < 100:
            #         probs[anchor] = 1 - diff
            #         self.correct_pred += 1

        # cv2.putText(frame, "# ml used: "+ str(self.total_pred), (50, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255))
        # cv2.putText(frame, "# kalman needed: " + str(self.correct_pred), (50, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0))
        # cv2.putText(frame, "total ids: " + str(self.maxId - 1), (50, 110), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0))
        
        new_attributes = obj.get_attributes()

        if len(probs) == 0: # acquire
            self.anchors[self.maxId] = new_attributes
            self.maxId += 1
            id = self.maxId
        else:               # require
            id = max(probs, key=probs.get)
            self.anchors[id] = new_attributes

        obj.draw_id(id, frame)

    def match(self, obj, frame):
        if self.is_training:
            self.training_match(obj)
        else:
            self.model_match(obj, frame)

    def save_data(self, name="tracking_data", dir="data"):
        if not self.is_training or len(self.data) == 0:
            print("No training data could be saved!")
            return

        df = pd.DataFrame(self.data)
        df.columns = ['Position', 'Magnitude', 'Direction', 'Area', 'Rotation', 'class']

        path = os.path.join("..", dir)
        if not os.path.isdir(path):
            os.mkdir(path)

        df.to_csv(os.path.join(path, name + ".csv"), index=False)
