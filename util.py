import pandas as pd 
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def load_data(path):
	df = pd.read_csv(path)
	return df.iloc[:, 1:-1], df['target']

def onehot_encoder(y_train):
	le = LabelEncoder()
	y_train_encoded = le.fit_transform(y_train)
	return to_categorical(y_train_encoded)

# def feature_ranking(X, y):

# 	forest = ExtraTreesClassifier(n_estimators=250,
# 	                              random_state=0)

# 	forest.fit(X, y)
# 	importances = forest.feature_importances_
# 	indices = np.argsort(importances)[::-1]

# 	# Print the feature ranking
# 	print("Feature ranking:")

# 	for f in range(X.shape[1]):
# 	    print("%d. feature %d (%.4f)" % (f + 1, indices[f], importances[indices[f]]))


# def plot_explained_var(k, X):
# 	svd = TruncatedSVD(n_components=k, n_iter=10)
# 	X_selected = svd.fit_transform(X)
# 	var_exp = svd.explained_variance_ratio_
# 	cum_var_exp = np.cumsum(var_exp)

# 	with plt.style.context('seaborn-whitegrid'):
# 	    plt.bar(range(20), var_exp, alpha=0.5, align='center',
# 	            label='individual explained variance')
# 	    plt.step(range(20), cum_var_exp, where='mid',
# 	             label='cumulative explained variance')
# 	    plt.ylabel('Explained variance ratio')
# 	    plt.xlabel('Principal components')
# 	    plt.legend(loc='best')
# 	    plt.tight_layout()