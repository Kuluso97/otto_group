import numpy as np 
import pandas as pd
from model.gdbt import GradientBoosting
from model.logreg import LogReg
from model.nn import NN 
from model.random_forest import RandomForest
from util import load_data
from sklearn.model_selection import train_test_split

DATA_PATH = './data/train.csv'

def main():
	X, y = load_data(DATA_PATH)
	X_train, X_test, y_train, y_test = train_test_split(
								X, y, test_size=0.1, random_state=1)

	models = [GradientBoosting, LogReg, NN, RandomForest]

	datadict = {}
	probs = []
	for Model in models:
		clf = Model()
		print("Running Model %s: " %clf.name)
		clf.fit(X_train, y_train)

		clf_name = clf.name
		pred_class = clf.predict_class(X_test, label=False)
		pred_proba = clf.predict_proba(X_test, normalize=True)
		datadict[clf_name] = pred_class
		probs.append(pred_proba)

	df = pd.DataFrame(data=datadict)
	df.to_csv("./data/engineered_feature.csv")


if __name__ == '__main__':
	main()
