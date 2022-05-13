
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
from matplotlib import pyplot as plt
import statsmodels.api as sm
import joblib
import pandas as pd
import glob
import os
import seaborn as sns


DATA_PATH = "../data/all_tracking.csv"
DATA_SPLIT = 0.01


def save(model, model_name, dir="model", ext=".pkl"):
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


def fit(model, model_name, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)

    save(model, model_name)

    y_pred = model.predict(x_test)

    print(model_name)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    print("F1 Score:", metrics.f1_score(y_test, y_pred), "\n")
    print("Confusion Matrix: \n", metrics.confusion_matrix(y_test, y_pred), "\n")
    print("Classification Report: \n", metrics.classification_report(y_test, y_pred))

    if model_name == "reg":
        X2 = sm.add_constant(x_train)
        est = sm.OLS(y_train, X2)
        est2 = est.fit()
        print(est2.summary())


def coor_plot(df):
    fig, ax = plt.subplots(figsize=[10, 50])
    sns.heatmap(df.corr(), cmap = 'coolwarm', square=True, ax=ax, 
                annot=True, linewidths=2, vmax=1, linecolor='white', 
                annot_kws={'fontsize': 13})
    plt.title('Feature Correlation', y=1.05, size=16)
    plt.show()


def train(y_col="class"):
    df = pd.read_csv(DATA_PATH)

    data = df.drop(y_col, axis=1)

    coor_plot(data)

    target = df[y_col]
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=DATA_SPLIT)

    fit(GaussianNB(), "bayes", x_train, y_train, x_test, y_test)
    fit(KNeighborsClassifier(n_neighbors=3), "knn", x_train, y_train, x_test, y_test)
    #fit(SVC(kernel="linear"), "svm", x_train, y_train, x_test, y_test)
    fit(LogisticRegression(max_iter=1000), "reg", x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    train()
    #concat_csv()