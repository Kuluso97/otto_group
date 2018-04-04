import numpy as np 
import pandas as pd 
from util import load_data, onehot_encoder
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

def onehot_encoder(y_train):
	le = LabelEncoder()
	y_train_encoded = le.fit_transform(y_train)
	return to_categorical(y_train_encoded)

def main():
	DATA_PATH = './data/train.csv'
	X, y = load_data(DATA_PATH)

	X_train, X_test, y_train, y_test = train_test_split(
					X, y, test_size=0.1, random_state=1)

	y_train_onehot = onehot_encoder(y_train)

	model = Sequential()
	model.add(Dense(
				units=93, 
				input_dim=X_train.shape[1],
				activation='relu'))
	model.add(Dense(
				units=9, 
				input_dim=X_train.shape[1],
				activation='softmax'))

	sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)
	model.compile(
				optimizer=sgd,
				loss='categorical_crossentropy')

	history = model.fit(X_train, y_train_onehot, batch_size=64, epochs=60,
						verbose=1, validation_split=0.1)

	y_test_pred = model.predict(X_test, verbose=0)

	loss = log_loss(y_pred=y_test_pred, y_true=y_test)
	print(loss)
	model.save('./models/nn.h5')

if __name__ == '__main__':
	main()