import xgboost as xgb
import numpy as np
from util import load_data, BaseClassifier

class GradientBoosting(BaseClassifier):

	def __init__(self, max_depth=15, eta=.1, silent=1, objective='multi:softprob',
					num_class=9, min_child_weight=5, colsample_bytree=45. / 93, 
					subsample=.8, nthread=8, seed=1, num_round=260):

		param = {'bst:max_depth':max_depth, 'bst:eta':eta, 'silent':silent, 
				'objective':objective,'num_class':num_class,
				'min_child_weight':min_child_weight, 
				'subsample':subsample, 'colsample_bytree':colsample_bytree, 
				'nthread':nthread, 'seed':seed}

		super().__init__()
		self.plst = param.items()
		self.num_round = num_round
		self.name = 'gdbt'

	def fit(self, X_train, y_train):
		y_train_encoded = self.le.fit_transform(y_train)
		dtrain = xgb.DMatrix(X_train, label=y_train_encoded)
		bst = xgb.train(self.plst, dtrain, self.num_round)
		self.clf = bst

	def predict_class(self, X_test, label=False):
		probs = self.predict_proba(X_test)
		classes_ = np.argmax(probs, axis=1)
		if label:
			classes_ = self.le.inverse_transform(classes_)
		return classes_

	def predict_proba(self, X_test, normalize=False):
		dtest = xgb.DMatrix(X_test)
		probas = self.clf.predict(dtest)

		if normalize:
			probas /= probas.sum(axis=1)[:, np.newaxis]

		return probas

	def save_model(self, path='../trained_models/0001.model'):
		if not self.clf:
			print("No model trained")
		else:
			self.clf.save_model(path)
			