from util import load_data, BaseClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

class SVM(BaseClassifier):

	def __init__(self, clf):
		super().__init__(clf)
		self.name = 'SVM'

def main():
	DATA_PATH = './data/train.csv'
	X, y = load_data(DATA_PATH)
	X_train, X_test, y_train, y_test = train_test_split(
								X, y, test_size=0.1, random_state=1)
	clf = SVC(probability=True)
	svm = SVM(clf)
	svm.fit(X_train, y_train)
	pred = svm.predict_class(X_test)
	print(pred)

if __name__ == '__main__':
	main()

