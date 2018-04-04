import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

class BaseClassifier(object):

	def __init__(self, clf):
		self.clf = clf
		self.le = LabelEncoder()

	def fit(self, X_train, y_train):
		"""
			Transform y_train from 'Class_i' to integer number
			in range [0,8].
		"""
		y_train_encoded = self.le.fit_transform(y_train)
		self.clf.fit(X_train, y_train_encoded)

	def predict_class(self, X_test, label=False):
		"""
			Return predicted classes

			Parameters
			----------

			label: Boolean

			if True, return String type label
			if False, return Integer type label

		"""
		classes_ = self.clf.predict(X_test)
		if label:
			classes_ = self.le.inverse_transform(classes_)
		return classes_

	def predict_proba(self, X_test, normalize=False):
		probas = self.clf.predict_proba(X_test)

		if normalize:
			probas /= probas.sum(axis=1)[:, np.newaxis]

		return probas

	def score(self, X_test, y_test):
		test_pred = self.predict_proba(X_test, normalize=True)
		return log_loss(y_pred=test_pred, y_true=y_test, eps=1e-15, normalize=True)

	def write_result(self, outputFile='submission.csv'):
		classes_ = ['Class_{}'.format(i) for i in range(1,10)]
		df = pd.read_csv('./data/test.csv')
		ids = pd.DataFrame(df["id"].values,columns= ["id"])
		array = pd.DataFrame(test_pred, columns=classes_)
		complete_array = pd.concat([ids,array],axis=1)
		complete_array.to_csv(outputFile ,sep=",", index=None)


def load_data(path):
	df = pd.read_csv(path)
	if 'target' in df.columns:
		return df.iloc[:, 1:-1], df['target']
	else:
		return df.iloc[:, 1:]