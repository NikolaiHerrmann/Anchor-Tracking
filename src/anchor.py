
from color import Color

class Anchor:

    def __init__(self):
        self.anchors = [None] * len(Color)
        self.data = []

    def match_generate_data(self, obj):
        anchored = False

        for anchor in self.anchors:
            if not anchor:
                continue
            
            thresholds = obj.compare(anchor)
            match = obj.compare_truth(anchor)
            if match:
                anchored = True

            thresholds.append(1 if match else 0)
            self.data.append(thresholds)

            # save data

            ## possibly remove anchor if old

        if not anchored:
            self.anchors[obj.color.value] = obj

    def match_ml(self):
        pass

    def save_data(self):
        for x in self.data:
            print(x)
