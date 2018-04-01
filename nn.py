import numpy as np 
import pandas as pd 
import keras
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('train.csv')

y = df['target']
X = df.iloc[:, 1:-1]

X_train, X_test, y_train, y_test = train_test_split(
				X, y, test_size=0.1, random_state=1)

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_train_onehot = keras.utils.to_categorical(y_train_encoded)
print(y_train_onehot.shape)

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

history = model.fit(X_train, y_train_onehot, batch_size=64, epochs=50,
					verbose=1, validation_split=0.1)

y_test_pred = model.predict_classes(X_test,
									verbose=0)
y_test_pred = le.inverse_transform(y_test_pred)

print(y_test_pred)
correct_preds = np.sum(y_test_pred == y_test)
test_acc = correct_preds / y_test.shape[0]

print("Test accuracy: %.2f%%" %(test_acc * 100))



