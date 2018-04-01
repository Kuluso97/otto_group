import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('train.csv')

y = df['target']
X = df.iloc[:, 1:-1]

X_train, X_test, y_train, y_test = train_test_split(
				X, y, test_size=0.1, random_state=1)

print(X_train.shape, y_train.shape)

## Build models (LogisticRegression):

clf = LogisticRegression()
clf.fit(X_train, y_train)
print('LogisticRegression LogLoss {score}'.format(
				score=log_loss(y_test, clf.predict_proba(X_test))))

print('Accuracy Rate: %.2f' %clf.score(X_test, y_test))








