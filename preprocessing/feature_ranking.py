import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from util import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.ensemble import ExtraTreesClassifier

DATA_PATH = '../data/train.csv'
X, y = load_data(DATA_PATH)
df = pd.read_csv(DATA_PATH)

forest = ExtraTreesClassifier(n_estimators=50, random_state=1)
forest.fit(X,y)

feat_labels = df.columns[1:-1]
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

for f in range(X.shape[1]):
	print("%2d) %-*s %f" % (f + 1, 30, 
							feat_labels[indices[f]], 
							importances[indices[f]]))

plt.title('Feature Importance')
plt.bar(range(30), 
		importances[indices][:30],
		align='center')

plt.xticks(range(30), 
			feat_labels[indices][:30], rotation=90)

plt.xlim([-1, 30])
plt.tight_layout()
plt.show()