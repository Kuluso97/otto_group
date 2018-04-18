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