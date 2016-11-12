from sklearn.metrics import classification_report
from split_training_test import test_file_name

import numpy as np

result_file = "../bytecup2016data/test_label_mahout.csv"

y_true = np.genfromtxt(test_file_name,delimiter=",", usecols=[2], skip_header=1)
y_pred = np.genfromtxt(result_file,delimiter=",", usecols=[2])

print classification_report(y_true, y_pred)

'''
TanimotoCoefficientSimilarity
             precision    recall  f1-score   support

        0.0       0.88      1.00      0.94     30351
        1.0       0.72      0.01      0.02      4065

avg / total       0.86      0.88      0.83     34416


UncenteredCosineSimilarity
             precision    recall  f1-score   support

        0.0       0.91      0.95      0.93     30351
        1.0       0.43      0.27      0.33      4065

avg / total       0.85      0.87      0.86     34416

'''