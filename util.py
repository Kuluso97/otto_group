import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def load_data(path):
	df = pd.read_csv(path)
	if 'target' in df.columns:
		return df.iloc[:, 1:-1], df['target']
	else:
		return df.iloc[:, 1:]

def write_result(path, test_pred):
	df = pd.read_csv(path)
	ids = pd.DataFrame(df["id"].values,columns= ["id"])
	array = pd.DataFrame(test_pred,
		columns=["Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"])
	complete_array = pd.concat([ids,array],axis=1)
	complete_array.to_csv("submission.csv",sep=",",index=None)

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