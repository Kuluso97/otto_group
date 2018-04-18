import numpy as np
from util import load_data, BaseClassifier
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical

class NN(BaseClassifier):

	def __init__(self, epochs=60, batch_size=64, validation_split=0.1):
		clf = Sequential()
		super().__init__(clf)
		self.name = 'nn'
		self.optimizer = SGD(lr=0.001, decay=1e-7, momentum=.9)
		self.epochs = epochs
		self.batch_size = batch_size
		self.validation_split = validation_split

	def fit(self, X_train, y_train):
		y_train_encoded = self.le.fit_transform(y_train)
		y_train_onehot = to_categorical(y_train_encoded)
		self.clf.add(Dense(
				units=93, 
				input_dim=X_train.shape[1],
				activation='relu'))
		self.clf.add(Dense(
				units=9, 
				input_dim=X_train.shape[1],
				activation='softmax'))

		self.clf.compile(				
				optimizer=self.optimizer,
				loss='categorical_crossentropy')

		self.clf.fit(X_train, y_train_onehot, batch_size=self.batch_size, epochs=self.epochs,
						verbose=1, validation_split=self.validation_split)

	def predict_class(self, X_test, label=False):
		probs = self.predict_proba(X_test)
		classes_ = np.argmax(probs, axis=1)
		if label:
			classes_ = self.le.inverse_transform(classes_)
		return classes_

	def predict_proba(self, X_test, normalize=False):
		probas = self.clf.predict(X_test)

		if normalize:
			probas /= probas.sum(axis=1)[:, np.newaxis]

		return probas

	def save_model(self, path='../trained_models/nn.h5'):
		self.clf.save(path)



