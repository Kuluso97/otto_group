from util import load_data, test_model
from sklearn.svm import SVC

DATA_PATH = './data/train.csv'

clf = SVC(probability=True)
loss = test_model(path=DATA_PATH, clf=clf)
print('SVM LogLoss {score}'.format(score=loss))
