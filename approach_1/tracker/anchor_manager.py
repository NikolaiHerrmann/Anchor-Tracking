
import numpy as np
import pandas as pd
import joblib
import os
import cv2
import sys
sys.path.append(os.path.join("..", "kalman_filter"))
from kf import KF



class AnchorManager:

    THRESHOLD_PROBA = 0.72

    def __init__(self, is_training):
        self.is_training = is_training
        self.anchors = {}
        self.ids_taken = set()


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
            #thresholds = obj.get_thresholds(anchor_attributes)

            #self.data.append(thresholds)

        #self.anchors[obj.color.value] = obj.get_attributes()

    def model_match(self, obj, frame, found):
        cv2.putText(frame, "# IDS " + str(self.maxId - 1), (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if not found:
            return
        
        probs = {}
        
        for anchor, anchor_attributes in self.anchors.items():

            if anchor in self.ids_taken:
                continue
                
            (x, y) = anchor_attributes[6].predict()

            #anchor_attributes[0] = np.array([x, y])
            #obj.draw_kalman_prediction(frame, x, y)

            thresholds = obj.get_thresholds(anchor_attributes)
            
        
            del thresholds[4]
            features = np.array([thresholds[0:4]])
            
            # target = thresholds[5]
            
            proba = self.model.predict_proba(features)[0]

            if proba[0] <= AnchorManager.THRESHOLD_PROBA:
                probs[anchor] = proba[1]
   
        new_attributes = obj.get_attributes()

        if len(probs) == 0: # acquire
            kal = KF()
            kal.predict()
            kal.update(obj.position[0], obj.position[1])
            new_attributes.append(kal)

            self.anchors[self.maxId] = new_attributes
            self.maxId += 1
            id = self.maxId
        else:               # re-acquire
            id = max(probs, key=probs.get)
            self.anchors[id][6].update(obj.position[0], obj.position[1])
            self.anchors[id] = new_attributes + [self.anchors[id][6]]

        self.ids_taken.add(id)
        obj.draw_id(id, frame)

    def reset_ids(self):
        self.ids_taken.clear()

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
