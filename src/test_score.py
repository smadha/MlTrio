from sklearn.metrics import classification_report
from split_training_test import test_file_name

import numpy as np

result_file = test_file_name

y_true = np.genfromtxt(test_file_name, usecols=[2])
y_pred = np.genfromtxt(test_file_name, usecols=[2])

print classification_report(y_true, y_pred)