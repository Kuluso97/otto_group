from util import load_data, BaseClassifier
from sklearn.ensemble import RandomForestClassifier

class RandomForest(BaseClassifier):

	def __init__(self):
		clf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
		super().__init__(clf)
		self.name = 'RandomForest'