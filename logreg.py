import numpy as np
import pandas as pd 
from util import load_data, test_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression

DATA_PATH = './data/train.csv'

clf = LogisticRegression()
loss = test_model(path=DATA_PATH, clf=clf)
print('Logistic Regression LogLoss {score}'.format(score=loss))








