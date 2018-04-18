from util import load_data, BaseClassifier
from sklearn.svm import SVC

class SVM(BaseClassifier):

	def __init__(self):
		clf = SVC(probability=True)
		super().__init__(clf)
		self.name = 'SVM'
