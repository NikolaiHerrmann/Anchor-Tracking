
class Anchor:

    def __init__(self):
        self.anchors = []

    def match_generate_data(self, obj):
        truth = 0

        for anchor in self.anchors:
            thresholds = obj.compare(anchor)
            truth |= obj.compare_truth(anchor)
            # save data

            ## possibly remove anchor if old

        if not truth:
            self.anchors.append(obj)

    def match_ml(self):
        pass

    def save_data(self):
        pass
