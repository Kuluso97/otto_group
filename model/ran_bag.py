from util import load_data, BaseClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split

DATA_PATH = '../data/train.csv'



class RandomForest(BaseClassifier):
	def __init__(self):
		clf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
		clfbag = BaggingClassifier(clf, n_estimators=5)
		super().__init__(clfbag)
		self.name = 'RandomForest'

def main():
	DATA_PATH = '../data/train.csv'
	X, y = load_data(DATA_PATH)
	X_train, X_test, y_train, y_test = train_test_split(
								X, y, test_size=0.1, random_state=1)

	forest = RandomForest()
	forest.fit(X_train, y_train)
	pred = forest.predict_class(X_test, label=True)
	print(pred)
	score = forest.score(X_test, y_test)
	print("The log loss of Random Forest model is: %.5f"  %score)


if __name__ == '__main__':
	main()
