
import numpy as np
import pandas as pd
import joblib
import os
import cv2


class AnchorManager:

    THRESHOLD_PROBA = 0.72

    def __init__(self, is_training, kalman_help):
        self.is_training = is_training
        self.kalman_help = kalman_help
        self.anchors = {}

        if not self.is_training:
            self.load_model("logistic_regression")
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

    def model_match(self, obj, frame, found):
        cv2.putText(frame, "# IDS " + str(self.maxId - 1), (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if self.kalman_help:
            x_, y_ = obj.draw_kalman_prediction(frame)

        if not found:
            return
        
        probs = {}
        
        for anchor, anchor_attributes in self.anchors.items():

            thresholds = obj.get_thresholds(anchor_attributes)
            del thresholds[4]
            #del thresholds[2]
            features = np.array([thresholds[0:4]])
            
            # target = thresholds[5]
            
            pred = self.model.predict(features)[0]
            proba = self.model.predict_proba(features)[0]

            # if pred == 1:
            #     probs[anchor] = proba[1]
            # elif proba[0] < 0.75:
            #     probs[anchor] = proba[0]
            #     print(proba)
            if proba[0] <= AnchorManager.THRESHOLD_PROBA:
                probs[anchor] = proba[1]

        
        # if self.kalman_help:
        #     if len(probs) == 0:
        #         for anchor, anchor_attributes in self.anchors.items():
        #             diff = np.linalg.norm(obj.position - anchor_attributes[0])
        #             if diff < 100:
        #                 probs[anchor] = 1 - diff


        # cv2.putText(frame, "# ml used: "+ str(self.total_pred), (50, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255))
        # cv2.putText(frame, "# kalman needed: " + str(self.correct_pred), (50, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0))
        
        new_attributes = obj.get_attributes()
        print(probs)

        if len(probs) == 0: # acquire
            self.anchors[self.maxId] = new_attributes
            self.maxId += 1
            id = self.maxId
        else:               # re-acquire
            id = max(probs, key=probs.get)
            self.anchors[id] = new_attributes

        obj.draw_id(id, frame)

    def match(self, obj, frame, found):
        if self.is_training:
            self.training_match(obj)
        else:
            self.model_match(obj, frame, found)

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
