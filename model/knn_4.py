from util import load_data, BaseClassifier
from sklearn.neighbors import KNeighborsClassifier

class KNN4(BaseClassifier):

	def __init__(self):
		clf = KNeighborsClassifier(n_neighbors=4)
		super().__init__(clf)
		self.name = 'knn_4'

