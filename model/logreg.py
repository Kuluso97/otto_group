from util import load_data, BaseClassifier
from sklearn.linear_model import LogisticRegression


class LogReg(BaseClassifier):

	def __init__(self):
		clf = LogisticRegression()
		super().__init__(clf)
		self.name = 'Logreg'
