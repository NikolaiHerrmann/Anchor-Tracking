
import os
import glob
from video_capture import VideoCapture
from ml import *
import matplotlib
import numpy as np
import sys


TRAIN_PATH = os.path.join("videos", "train")
TEST_PATH = os.path.join("videos", "test")
SHOW_GUI = True
RECORD = True


def train():
    for vid_file in glob.glob(os.path.join(TRAIN_PATH, "*.mp4")):
        print("Training Video: " + vid_file)
        video_capture = VideoCapture(vid_file, is_training=True, show_gui=SHOW_GUI)
        video_capture.read()


def test_single(model_name, vid_file, show_gui=SHOW_GUI):
    print("Testing Video: " + vid_file)
    video_capture = VideoCapture(vid_file, is_training=False, show_gui=show_gui, record=RECORD)
    video_capture.select_model(model_name)
    vid_acc = video_capture.read()
    print("Metrics: " + str(vid_acc))
    return vid_acc


def test(model_name):
    for diff in ["easy", "medium", "hard"]:
        acc = []
        print("Testing " + str(diff))
        for vid_file in glob.glob(os.path.join(TEST_PATH, diff, "*.mp4")):
            acc.append(test_single(model_name, vid_file, show_gui=SHOW_GUI))
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
    #train()
    #concat_csv("data")

    # data = get_data()

    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(17, 5))
    # ls = [ax1, ax2, ax3, ax4]

    # for name, model in MODELS.items():
    #     print("Testing " + name)
    #     fit_model(model, name, data, ls.pop(0))
    #     #test(name)
    # for ax in fig.get_axes():
    #     ax.set_ylabel("F1 Score", fontsize=14)

    # fig.text(0.5, 0.01, "Training Data Size", ha='center', fontsize=14)
    # plt.title("Learning Curves for Classifiers")

    # for ax in fig.get_axes():
    #     ax.label_outer()

    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, ncol=1,loc='right')#, bbox_to_anchor=(0.9, -0.01))

    # save_graph("main")
    # plt.show()

    # clean_up()

    # test("Logistic_Regression")
    #test_single("Logistic_Regression", "videos/train/tv3_blue_yellow_green_brown_pink.mp4")
    #test_single("Logistic_Regression", "videos/test/tv1_blue_pink.mp4")

    # for name, model in MODELS.items():
    #     print("Testing " + name)
    #     test(name)

    #test_single("Naive_Bayes", "videos/test/medium/m4_yellow_blue_green2_pink.mp4")

    # data = get_data(y_col="Target")
    # matplotlib.rcParams.update({'font.size': 10})

    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 5))
    # ls = [ax1, ax2, ax3, ax4]

    # for name, model in MODELS.items():
    #     print("Testing " + name)
    #     fit_model(model, name, data, ls.pop(0))
    #     #test(name)
    # for ax in fig.get_axes():
    #     ax.set_ylabel("F1 Score", fontsize=15, labelpad=10)

    # fig.text(0.5, 0.01, "Training Data Size", ha='center', fontsize=15)
    # plt.subplots_adjust(bottom=0.15, top=0.85, left=0.08, wspace=0.1)

    # for ax in fig.get_axes():
    #     ax.label_outer()

    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, ncol=1, loc='right', labelspacing = 1.5)
    # fig.set_size_inches(15, 5)
    # plt.savefig("a2_lc.png", dpi=1000)
    # fig.text(0.5, 0.96, "Approach 2 Learning Curves", ha='center', fontsize=15)
    # plt.savefig("a2_ls.pdf")
    # plt.show()

    # save_graph("main")
    # plt.show()

    test_single("Naive_Bayes", "videos/test/easy/tv10_brown_pink.mp4")