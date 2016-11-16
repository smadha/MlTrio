'''
Uses flattened features in feature directory and run a SVM on it
'''

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import cPickle as pickle

def normalize(X_tr):
    ''' Normalize training and test data features
    Args:
        X_tr: Unnormalized training features
    Output:
        X_tr: Normalized training features
    '''
    X_mu = np.mean(X_tr, axis=0)
    X_tr = X_tr - X_mu
    X_sig = np.std(X_tr, axis=0)
    X_tr = X_tr/X_sig
    return X_tr, X_mu, X_sig


features = pickle.load( open("../feature_engg/feature/all_features.p", "rb") )
labels = [int(l) for l in pickle.load( open("../feature_engg/feature/labels.p", "rb") )]


print len(features),len(features[0])
print len(labels),labels[0]

features = np.array(features)

col_deleted = np.nonzero((features==0).sum(axis=0) > (len(features)-1000))
print col_deleted
features = np.delete(features, col_deleted, axis=1)

print len(features),len(features[0])
print len(labels),labels[0]

features, X_mu, X_sig = normalize(features)

X_tr, X_te,y_tr, y_te = train_test_split(features,labels, train_size = 0.85)

clf = SVC(kernel='rbf',C=16,gamma=0.015625,class_weight={0: 0.75 , 1:1}, cache_size=1000, tol=1e-1)

clf.fit(X_tr,y_tr)


print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_te, clf.predict(X_te)
print(classification_report(y_true, y_pred))
print()

print "done"    
    
