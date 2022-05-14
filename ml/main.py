
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from matplotlib import pyplot as plt
import statsmodels.api as sm
import joblib
import pandas as pd
import glob
import os
import seaborn as sns
import numpy as np


DATA_PATH = os.path.join("..", "data", "all_tracking.csv")
DATA_SPLIT = 0.3
FOLDS = 5
KNN_N_NEIGHBORS = 3
GRAPH_DIR = "graphs"


def save(model, model_name, dir="model", ext=".pkl"):
    model_name = model_name.replace(" ", "_").lower()
    path = os.path.join("..", dir)
    if not os.path.isdir(path):
        os.mkdir(path)
    with open(os.path.join(path, model_name + ext), "wb") as f:
        joblib.dump(model, f)


def concat_csv(name="all_tracking"):
    dir = os.path.join("..", "data")
    files = os.path.join(dir, "tracking_data*.csv")
    files = glob.glob(files)
    print("Combining the following files:")
    print(files)
    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    df.to_csv(os.path.join(dir, name + ".csv"), index=False)


def save_graph(name, dir_=GRAPH_DIR):
    name = name.replace(" ", "_").lower()
    if not os.path.isdir(dir_):
        os.mkdir(dir_)
    plt.savefig(os.path.join(dir_, name + ".pdf"))


def plot_learning_curve(model, model_name, X, y, folds):
    """
    Adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    """
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=folds, n_jobs=None, return_times=False, scoring='f1_weighted'
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    _, axes = plt.subplots(figsize=(8, 5))
    axes.grid()
    axes.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="orange",
    )
    axes.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="purple",
    )
    axes.plot(train_sizes, train_scores_mean, "o-", color="orange", label="training")
    axes.plot(train_sizes, test_scores_mean, "o-", color="purple", label="cross-validation (" + str(folds) + " folds)")
    axes.set_xlabel("Number of Observations", fontsize=12)
    axes.set_ylabel("F1 Score", fontsize=12)
    axes.legend(loc="best")

    plt.title(model_name + " Learning Curve", fontsize=12)
    save_graph("l_curve_" + model_name)


def fit(model, model_name, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)

    save(model, model_name)

    y_pred = model.predict(x_test)

    acc = metrics.accuracy_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred)
    print(model_name, "\t", round(acc, 3), "\t\t", round(f1_score, 3))
    # print("Recall:", metrics.recall_score(y_test, y_pred))
    # print("Confusion Matrix: \n", metrics.confusion_matrix(y_test, y_pred), "\n")
    # print("Classification Report: \n", metrics.classification_report(y_test, y_pred))

    if model_name == "Logistic Regression":
        X2 = sm.add_constant(x_train)
        est = sm.OLS(y_train, X2)
        est2 = est.fit()
        print(est2.summary())


def coor_plot(df):
    _, ax = plt.subplots(figsize=(11, 8))
    plot = sns.heatmap(df.corr(), cmap='coolwarm', square=True, ax=ax,
                annot=True, linewidths=2, vmin = -0.2, vmax=1, linecolor='white',
                annot_kws={'fontsize': 13})
    plot.set_xticklabels(plot.get_xmajorticklabels(), fontsize=15)
    plot.set_yticklabels(plot.get_ymajorticklabels(), fontsize=15, va='center')
    plt.title('Feature Correlation', y=1.05, size=18)
    save_graph("feat_coor")


def get_data(path, y_col="class"):
    df = pd.read_csv(path)
    #df = np.array_split(df, 12)[0] # for quick testing
    X = df.drop(y_col, axis=1)
    y = df[y_col]
    return X, y


def train(X, y, split, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, shuffle=True, random_state=seed)

    models = {"Naive Bayes": GaussianNB(),
              "KNN": KNeighborsClassifier(n_neighbors=KNN_N_NEIGHBORS),
              "Logistic Regression": LogisticRegression(max_iter=1000)}

    print("\t Accuracy \t F1-Score")
    for name, model in models.items():
        fit(model, name, X_train, y_train, X_test, y_test)
        plot_learning_curve(model, name, X, y, FOLDS)


if __name__ == "__main__":
    # concat_csv()

    X, y = get_data(DATA_PATH)
    coor_plot(X)

    train(X, y, DATA_SPLIT)
