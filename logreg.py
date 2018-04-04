import numpy as np
import pandas as pd 
from util import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression

DATA_PATH = './data/train.csv'
X, y = load_data(DATA_PATH)

X_train, X_test, y_train, y_test = train_test_split(
				X, y, test_size=0.1, random_state=1)

## Build models (LogisticRegression):

clf = LogisticRegression()
clf.fit(X_train, y_train)
print('LogisticRegression LogLoss {score}'.format(
				score=log_loss(y_pred=clf.predict_proba(X_test), y_true=y_test)))

print('Accuracy Rate: %.2f' %clf.score(X_test, y_test))








