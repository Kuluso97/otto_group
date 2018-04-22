import numpy as np 
import pandas as pd
from model.knn_4 import KNN4
from model.knn_8 import KNN8
from model.knn_12 import KNN12
from model.gdbt import GradientBoosting
from util import load_data
from sklearn.model_selection import train_test_split

DATA_PATH = '../data/train.csv'
X, y = load_data(DATA_PATH)

X, y = load_data(DATA_PATH)
X_train, X_test, y_train, y_test = train_test_split(
							X, y, test_size=0.1, random_state=1)

models = [GradientBoosting()]

for clf in models:
	print("Running Model %s: " %clf.name)
	clf.fit(X_train, y_train)

	score = clf.score(X_test, y_test)
	print("The log loss of %s is: %s" %(clf.name, score))
