import numpy as np 
import pandas as pd
from model.gdbt import GradientBoosting
from model.logreg import LogReg
from model.nn import NN 
from model.random_forest import RandomForest
from util import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import sys

DATA_PATH = './data/train.csv'

class BaggingClassifier(object):

	def __init__(self, base_learners=None):
		self.base_learners = base_learners
		self.name='bagging'

	def _bootstrap(self, data):
		n_samples = data.shape[0]
		indices = np.random.randint(low=0, high=n_samples, size=(n_samples,))
		resampled_arrays = [data[i] for i in indices]
		return np.array(resampled_arrays)

	def fit(self, X_train, y_train):
		data = np.hstack((X_train, y_train[:,np.newaxis]))
		for clf in self.base_learners:
			print("Running Model %s: " %clf.name)
			sample = self._bootstrap(data)
			X_train, y_train = sample[:,:-1], sample[:,-1]
			clf.fit(X_train, y_train)

	def predict_proba(self, X_test, normalize=False):
		probs = []
		for learner in self.base_learners:
			pred = learner.predict_proba(X_test, normalize=True)
			probs.append(pred)

		probs = np.mean(probs, axis=0)
		if normalize:
			row_sum = np.sum(probs, axis=1)
			probs /= row_sum[:, np.newaxis]

		return probs

	def score(self, X_test, y_test):
		test_pred = self.predict_proba(X_test, normalize=True)
		return log_loss(y_pred=test_pred, y_true=y_test, eps=1e-15, normalize=True)

def main():
	X, y = load_data(DATA_PATH)
	X_train, X_test, y_train, y_test = train_test_split(
								X, y, test_size=0.1, random_state=1)

	models = [NN(), NN(), NN(), NN(), NN()]
	clf = BaggingClassifier(base_learners=models)
	clf.fit(X_train, y_train)
	probs = clf.predict_proba(X_test, normalize=True)
	print(probs.shape)
	score = clf.score(X_test, y_test)
	print("The log loss of Bagging Classifier is: %s" %score)

if __name__ == '__main__':
	main()



