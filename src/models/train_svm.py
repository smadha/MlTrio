'''
Uses flattened features in feature directory and run a SVM on it
'''

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
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
labels = pickle.load( open("../feature_engg/feature/labels.p", "rb") )


print len(features),len(features[0])
print len(labels),labels[0]

features = np.array(features)

col_deleted = np.nonzero((features==0).sum(axis=0) > (len(features)-1000))
print col_deleted
features = np.delete(features, col_deleted, axis=1)

print len(features),len(features[0])
print len(labels),len(labels[0])

features, X_mu, X_sig = normalize(features)

gamma_ramge = [ 4**i for i in range(-7,0) ]
C_range = [ 4**i for i in range(-3,6) ]

# Number of folds in Cross validation
CV_FOLDS = 3
# Number of parallel jobs
parallel = 6

X_tr, X_te,y_tr, y_te = train_test_split(features,labels, train_size = 0.85)

svr = SVC()

parameters = [{ 'kernel':['rbf'], 'C':C_range, 'gamma':gamma_ramge}]

clf = GridSearchCV(svr, parameters, cv=CV_FOLDS, n_jobs = parallel, verbose=10000)

clf.fit(X_tr,y_tr)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_te, clf.predict(X_te)
print(classification_report(y_true, y_pred))
print()

print "done"    
    
