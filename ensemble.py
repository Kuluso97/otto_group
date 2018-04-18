import numpy as np 
import pandas as pd 
from model.gdbt import GradientBoosting
from util import load_data
from sklearn.model_selection import train_test_split

DATA_PATH = './data/engineered_feature.csv'

def main():
	X, y = load_data(DATA_PATH)

	X_train, X_test, y_train, y_test = train_test_split(
								X, y, test_size=0.1, random_state=1)

	lg = LogReg()
	lg.fit(X_train, y_train)
	pred = lg.predict_class(X_test, label=True)
	print(pred)
	print(sum(pred == y_test) / y_test.shape[0])
	score = lg.score(X_test, y_test)
	print("The log loss of Ensemble model is: %.5f"  %score)

if __name__ == '__main__':
	main()