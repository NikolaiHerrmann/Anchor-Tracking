
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from matplotlib import pyplot as plt
import statsmodels.api as sm
import joblib
import pandas as pd
import glob
import os
import seaborn as sns
import numpy as np

DATA_DIR = "data"
DATA_SPLIT = 0.3
FOLDS = 5
KNN_N_NEIGHBORS = 3
GRAPH_DIR = "graphs"
THRESH_PLOT_OB = 100
CONCAT_FILE = "tv_all.csv"

MODELS = {"Naive_Bayes": GaussianNB(),
          "KNN": KNeighborsClassifier(n_neighbors=3),
          "Logistic_Regression": LogisticRegression(max_iter=1000),
          "Decision_Tree": RandomForestClassifier(max_depth=10)}


def get_save_path(name, dir, ext):
    name = name.replace(" ", "_")
    if not os.path.isdir(dir):
        os.mkdir(dir)
    return os.path.join(dir, name + ext)


def save_model(model, model_name, dir="models", ext=".pkl"):
    path = get_save_path(model_name, os.path.join(dir), ext)
    with open(path, "wb") as f:
        joblib.dump(model, f)


def save_graph(name, dir=GRAPH_DIR):
    path = get_save_path(name, dir, ".pdf")
    plt.savefig(path)


def concat_csv(dir, output_name="tv_all.csv"):
    files = os.path.join(dir, "tv*.csv")
    files = glob.glob(files)
    print("Combining the following files:")
    print(files)
    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    df.to_csv(os.path.join(dir, output_name), index=False)
    return output_name


def plot_learning_curve(model, model_name, X, y, folds):
    """
    Adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    """
    #sizes = [x for x in range(5000, np.intp(np.floor(len(y) * 0.8)), 5000)]

    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=folds, n_jobs=None,
        #train_sizes=sizes,
        return_times=False, scoring='f1_weighted'
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
        color="darkorange",
    )
    axes.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="purple",
    )
    axes.plot(train_sizes, train_scores_mean, "o-",
              color="darkorange", label="training")
    axes.plot(train_sizes, test_scores_mean, "o-", color="purple",
              label="cross-validation (" + str(folds) + " folds)")
    axes.set_xlabel("Number of Observations", fontsize=12)
    axes.set_ylabel("F1 Score", fontsize=12)
    axes.legend(loc="best")

    plt.title(model_name + " Learning Curve", fontsize=12)
    save_graph("l_curve_" + model_name)


def regression_stats(model, x_train, y_train):
    X_ = sm.add_constant(x_train)
    est = sm.OLS(y_train, X_)
    est_ = est.fit()
    print(est_.summary())


def tree_stats(model, x_train, y_train):
    importances = model.feature_importances_
    std = np.std([importances for _ in model.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=x_train.columns)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax, color="purple")
    ax.set_title("Feature Importances")
    ax.set_ylabel("Mean Decrease in Impurity")
    fig.tight_layout()
    save_graph("feat_import_tree")


def fit(model, model_name, x_train, y_train, x_test, y_test):
    model.fit(x_train.values, y_train.values)
    save_model(model, model_name)
    y_pred = model.predict(x_test.values)

    acc = metrics.accuracy_score(y_test.values, y_pred)
    f1_score = metrics.f1_score(y_test.values, y_pred)
    print(model_name, "\t", round(acc, 3), "\t\t", round(f1_score, 3))

    #find_thresh(model, x_train, y_train)

    if model_name == "Logistic Regression":
        regression_stats(model, x_train, y_train)
    if model_name == "Decision Tree":
        tree_stats(model, x_train, y_train)
        

def coor_plot(df):
    _, ax = plt.subplots(figsize=(11, 8))
    plot = sns.heatmap(df.corr(), cmap='coolwarm', square=True, ax=ax,
                       annot=True, linewidths=2, vmin=-0.2, vmax=1, linecolor='white',
                       annot_kws={'fontsize': 13})
    plot.set_xticklabels(plot.get_xmajorticklabels(), fontsize=15)
    plot.set_yticklabels(plot.get_ymajorticklabels(), fontsize=15, va='center')
    plt.title('Feature Correlation', y=1.05, size=18)
    save_graph("feat_coor")


def find_thresh(model, X_train, y_train, num_observations=THRESH_PLOT_OB):
    y_scores = model.predict_proba(X_train.values)[:, 1]
    threshold = []
    accuracy = []

    for t in range(0, num_observations + 1):
        t = t / num_observations
        threshold.append(t)
        y_pred_with_threshold = (y_scores >= t).astype(int)
        accuracy.append(metrics.balanced_accuracy_score(y_train, y_pred_with_threshold))

    max_thresh = threshold[np.argmax(accuracy)]
    print("Optimal Threshold: ", max_thresh)

    plt.clf()
    plt.ylim([0.87, 1])
    plt.vlines(x=max_thresh, ymin=0.87, ymax=1, colors="red",
               linestyle='dashed',
               label="Optimal Threshold (" + str(round(max_thresh, 3)) + ")")
    plt.legend(loc="best")
    plt.scatter(threshold, accuracy)
    plt.xlabel("Threshold", fontsize=14)
    plt.ylabel("Balanced Accuracy", fontsize=14)
    save_graph("optimal_thresh")


def undersample(df, y_col):
    min_count = df[y_col].value_counts().min()
    return df.groupby(y_col).apply(lambda x: x.sample(min_count)).reset_index(drop=True)


def shuffle(df):
    return df.sample(frac=1).reset_index(drop=True)


def check_balance(y):
    print("Data Set Balance: ")
    size = len(y)
    print("# observations =", size)
    print("% of 1's =", round((y == 1).sum() / size, 3))
    print("% of 0's =", round((y == 0).sum() / size, 3))


def get_data(path=CONCAT_FILE, y_col="Target"):
    df = pd.read_csv(path)

    check_balance(df[y_col])

    df = undersample(df, y_col)
    df = shuffle(df)

    X = df.drop(y_col, axis=1)
    y = df[y_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=DATA_SPLIT,
                                                        shuffle=True, random_state=45)

    return X, y, X_train, X_test, y_train, y_test


def fit_model(model, model_name, data):
    X, y, X_train, X_test, y_train, y_test = data
    print("\t Accuracy \t F1-Score")
    fit(model, model_name, X_train, y_train, X_test, y_test)
    plot_learning_curve(model, model_name, X, y, FOLDS)


def clean_up():
    os.remove(CONCAT_FILE)
