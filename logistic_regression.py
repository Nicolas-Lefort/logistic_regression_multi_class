# data from https://archive.ics.uci.edu/ml/datasets/smartphone-based+recognition+of+human+activities+and+postural+transitions
# Smartphone measurements vs human activities

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from prepare_data import process

# import data
df = pd.read_csv('smartphones_data.csv')
# define target feature
target = "Activity"
# prepare the data
df, df_numeric, _, _, _ = process(df=df, ordinal_features=None, target=target)
# define x and y
X, y = df.drop(columns=target), df[target]
# split train test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
# define the model
lr = LogisticRegression(solver='liblinear')
# fit to train data
lr = lr.fit(X_train, y_train)
# predict test data
y_pred = lr.predict(X_test)
# Precision, recall, f-score from the multi-class support function
print(classification_report(y_test, y_pred))
print('Accuracy score: ', round(accuracy_score(y_test, y_pred), 2))
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_train))
disp.plot(cmap=plt.cm.Blues)
plt.show()