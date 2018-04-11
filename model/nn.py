import numpy as np
from util import load_data, BaseClassifier
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class Model(BaseClassifier):

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


def main():
	DATA_PATH = '../data/train.csv'
	X, y = load_data(DATA_PATH)

	X_train, X_test, y_train, y_test = train_test_split(
					X, y, test_size=0.1, random_state=1)

	nn = Model()
	nn.fit(X_train, y_train)
	pred = nn.predict_class(X_test, label=True)
	print(pred)
	score = nn.score(X_test, y_test)
	print("The log loss of Neural Network model is: %.5f"  %score)
	nn.save_model()

if __name__ == '__main__':
	main()


