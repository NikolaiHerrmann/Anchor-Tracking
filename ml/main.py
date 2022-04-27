
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import os


df = pd.read_csv(os.path.join("..", "data", "tracking_data.csv"))

train, test = train_test_split(df, test_size=0.3)

y_train = train['same_object'].to_numpy()
del train['same_object']
x_train = train.to_numpy()

y_test = test['same_object'].to_numpy()
del test['same_object']
x_test = test.to_numpy()

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(y_train)

range_k = range(1, 15)
scores = {}
scores_list = []
for k in range_k:
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    scores[k] = metrics.accuracy_score(y_test, y_pred)
    scores_list.append(metrics.accuracy_score(y_test, y_pred))
result = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = metrics.classification_report(y_test, y_pred)
print("Classification Report:",)
print(result1)

plt.plot(range_k, scores_list)
plt.xlabel("Value of K")
plt.ylabel("Accuracy")
plt.show()
