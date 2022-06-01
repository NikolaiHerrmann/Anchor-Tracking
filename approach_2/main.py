
import os
import glob
from video_capture import VideoCapture
from ml import *


TRAIN_PATH = os.path.join("videos", "train")
TEST_PATH = os.path.join("videos", "test")
SHOW_GUI = False
RECORD = False


def train():
    for vid_file in glob.glob(os.path.join(TRAIN_PATH, "*.mp4")):
        print("Training Video: " + vid_file)
        video_capture = VideoCapture(vid_file, is_training=True, show_gui=SHOW_GUI)
        video_capture.read()


def test_single(model_name, vid_file):
    print("Testing Video: " + vid_file)
    video_capture = VideoCapture(vid_file, is_training=False, show_gui=SHOW_GUI, record=RECORD)
    video_capture.select_model(model_name)
    vid_acc = video_capture.read()
    print("Accuracy: " + str(vid_acc))
    return vid_acc


def test(model_name):
    acc = []

    for vid_file in glob.glob(os.path.join(TEST_PATH, "*.mp4")):
        acc.append(test_single(model_name, vid_file))

    print(acc)
    print(sum(acc) / len(acc))


if __name__ == "__main__":
    #train()
    #concat_csv("data")

    # data = get_data()

    # for name, model in MODELS.items():
    #     print("Testing " + name)
    #     fit_model(model, name, data)
    #     test(name)

    # clean_up()

    test("Logistic_Regression")
    #test_single("Logistic_Regression", "videos/train/tv5_blue_yellow_green_brown_pink.mp4")
