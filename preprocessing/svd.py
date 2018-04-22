import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

from util import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import TruncatedSVD

DATA_PATH = '../data/train.csv'
X, y = load_data(DATA_PATH)
df = pd.read_csv(DATA_PATH)

svd = TruncatedSVD(n_components=50, n_iter=10)
X_selected = svd.fit_transform(X)
var_exp = svd.explained_variance_ratio_
cum_var_exp = np.cumsum(var_exp)

with plt.style.context('seaborn-whitegrid'):
	plt.bar(range(50), var_exp, alpha=0.5, align='center',
			label='individual explained variance')
	plt.step(range(50), cum_var_exp, where='mid',
			label='cumulative explained variance')
	plt.ylabel('Explained variance ratio')
	plt.xlabel('Principal components')
	plt.legend(loc='best')
	plt.tight_layout()

plt.show()