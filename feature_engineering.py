import numpy as np 
import pandas as pd
from model.gdbt import GradientBoosting
from model.logreg import LogReg
from model.nn import NN 
from model.ran_bag import RandomForest
from model.ran_Calib import CalibratedRandomForest
from util import load_data
from sklearn.model_selection import train_test_split

DATA_PATH = './data/train.csv'

def main():
	X, y = load_data(DATA_PATH)
	X_train, X_test, y_train, y_test = train_test_split(
								X, y, test_size=0.1, random_state=1)

	models = [GradientBoosting(), LogReg(), NN(), RandomForest(), CalibratedRandomForest()]

	datadict = {}
	for clf in models:
		print("Running Model %s: " %clf.name)
		clf.fit(X_train, y_train)

		clf_name = clf.name
		pred_class = clf.predict_class(X, label=False)
		datadict[clf_name] = pred_class

	df = pd.DataFrame(data=datadict)
	df['target'] = pd.Series(np.array(y), index=df.index)
	df.to_csv("./data/engineered_feature.csv")


if __name__ == '__main__':
	main()
