from util import load_data, BaseClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

DATA_PATH = '../data/train.csv'

class CalibratedRandomForest(BaseClassifier):

	def __init__(self):
		clf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
		calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv=5)
		super().__init__(calibrated_clf)
		self.name = 'Calibrated RandomForest'


