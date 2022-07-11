
import os
from tracker import Tracker
import glob
import numpy as np
import sys


VID_PATH = "../../approach_2/videos"
DATA_PATH = "../data"
RECORD = True
SHOW_GUI = True


def train():
    path = os.path.join(VID_PATH, "train")
    for vid_file in glob.glob(os.path.join(path, "*.mp4")):
        print("Training Video: " + vid_file)
        tracker = Tracker(vid_file, DATA_PATH, is_training=True, record=RECORD, show_gui=SHOW_GUI)
        tracker.run()


def test_single(model_name, vid_file):
    print("Testing Video: " + vid_file)
    video_capture = Tracker(vid_file, DATA_PATH, is_training=False, record=RECORD, show_gui=SHOW_GUI)
    video_capture.select_model(model_name)
    vid_acc = video_capture.run()
    print("Metrics: " + str(vid_acc))
    return vid_acc


def test(model_name):
    for diff in ["easy", "medium", "hard"]:
        print("Testing on " + diff)
        acc = []
        for vid_file in glob.glob(os.path.join(os.path.join(VID_PATH, "test", diff), "*.mp4")):
            acc.append(test_single(model_name, vid_file))
            sys.stdout.flush()

        col_totals = [sum(x) / len(x) for x in zip(*acc)]
        f1_scores = [x[3] for x in acc]
        print(f1_scores)

        print("====================================")
        print("Results for " + str(model_name))
        print("Recall: ", col_totals[0])
        print("Precision: ", col_totals[1])
        print("Accuracy: ", col_totals[2])
        print("F1 Score: ", col_totals[3])
        print("F1 Score STD: ", np.std(f1_scores))
        print("====================================")


if __name__ == "__main__":

    # for name in ["Naive_Bayes", "KNN", "Logistic_Regression", "Decision_Tree"]:
    #     print("Testing " + name)
    #     test(name)

    test_single("Naive_Bayes", "../../approach_2/videos/test/easy/tv10_brown_pink.mp4")
