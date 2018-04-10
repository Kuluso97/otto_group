from util import load_data, BaseClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

DATA_PATH = './data/train.csv'

class LogReg(BaseClassifier):

	def __init__(self):
		clf = LogisticRegression()
		super().__init__(clf)
		self.name = 'Logreg'

def main():
	DATA_PATH = '../data/train.csv'
	X, y = load_data(DATA_PATH)
	X_train, X_test, y_train, y_test = train_test_split(
								X, y, test_size=0.1, random_state=1)

	logreg = LogReg()
	logreg.fit(X_train, y_train)
	pred = logreg.predict_class(X_test, label=True)
	print(pred)
	score = logreg.score(X_test, y_test)
	print("The log loss of Logistic Regression model is: %.5f"  %score)
	logreg.write_result()


if __name__ == '__main__':
	main()
