
from color import Color
import pandas as pd
import os

class Anchor:

    def __init__(self):
        self.anchors = [None] * len(Color)
        self.data = []

    def match_generate_data(self, obj):

        for anchor_attributes in self.anchors:
            if not anchor_attributes:
                continue
            
            thresholds = obj.get_thresholds(anchor_attributes)

            self.data.append(thresholds)

            ## possibly remove anchor if old

        self.anchors[obj.color.value] = obj.get_attributes()

    def match_ml(self):
        pass

    def save_data(self, dir = "data", name = "tracking_data.csv"):
        df = pd.DataFrame(self.data)
        df.columns = ['position_thresh', 'magnitude_thresh', 'direction_thresh', 'area_thresh', 'rotation_thresh', 'same_object']
        os.chdir("..")
        if not os.path.isdir(dir):
            os.mkdir(dir)
        df.to_csv(os.path.join(dir, name), index = False)
