from util import test_model, write_result
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = './data/train.csv'
TEST_PATH = './data/test.csv'

clf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv=5)

clf, loss = test_model(path=DATA_PATH, clf=clf)
print('Calibrated Classifier LogLoss {score}'.format(score=loss))

X_test = load_data(TEST_PATH)
test_pred = clf.predict_proba(X_test)

write_result(path=TEST_PATH, test_pred=test_pred)
