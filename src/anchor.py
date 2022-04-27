
from color import Color

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

    def save_data(self):
        for x in self.data:
            print(x)
