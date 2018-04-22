from util import load_data, BaseClassifier
from sklearn.neighbors import KNeighborsClassifier

class KNN8(BaseClassifier):

	def __init__(self):
		clf = KNeighborsClassifier(n_neighbors=8)
		super().__init__(clf)
		self.name = 'knn_8'