from keras.models import load_model
from util import write_result, load_data

TEST_PATH = './data/test.csv'

X_test = load_data(TEST_PATH)
model = load_model('./models/nn.h5')
test_pred = model.predict(X_test)

write_result(path=TEST_PATH, test_pred=test_pred)