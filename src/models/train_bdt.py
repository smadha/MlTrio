'''
Trains a boosting tree
'''
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import cPickle as pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, make_scorer


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


# features = pickle.load( open("../feature_engg/feature/all_features.p", "rb") )
# labels = [int(l) for l in pickle.load( open("../feature_engg/feature/labels.p", "rb") )]

features = np.random.normal(size=(2294,354))
labels = [0]*2000 + [1]*(2294-2000)

print len(features),len(features[0])
print len(labels),labels[0]

features = np.array(features)

col_deleted = np.nonzero((features==0).sum(axis=0) > (len(features)-1000))
print col_deleted
features = np.delete(features, col_deleted, axis=1)

print len(features),len(features[0])
print len(labels),labels[0]

features, X_mu, X_sig = normalize(features)
print "data normalised"

save_res = {"col_deleted":col_deleted,"X_mu":X_mu,"X_sig":X_sig}
with open("model/train_config_bdt.p", 'wb') as pickle_file:
    pickle.dump(save_res, pickle_file, protocol=2)
print "Dumped config"


max_tree_depth_range = [4]
num_estimators_range = [100]
learning_rate_range = [1]

def run_BDT(max_tree_depth, num_estimators,learning_rate, save=False, test=True):
    if test:
        features_tr, features_te,labels_tr, labels_te = train_test_split(features,labels, train_size = 0.85)
        print "Using separate test data"
    else:
        features_tr, features_te,labels_tr, labels_te = features,labels, features[0:1000],labels[0:1000]
        print "Using a sample training data"
        
    # Create and fit an AdaBoosted decision tree
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_tree_depth), n_estimators=num_estimators)
    
    bdt.fit(features_tr, labels_tr)
    
    err = bdt.estimator_errors_
    print "BDT estimator_errors_ max {0}, min {1}, avg {2}, std {3}".format(max(err),min(err),np.average(err),np.std(err))
    
    print("Classification report on sample/test data:")
    y_true, y_pred = labels_te, bdt.predict(features_te)
    print classification_report(y_true, y_pred)
    print "max_tree_depth, num_estimators, learning_rate",max_tree_depth, num_estimators, learning_rate
    
    if save:
        pickle.dump(bdt, open("model/model_bdt.h5","w"), protocol=2)
    
for max_tree_depth in max_tree_depth_range:
    for num_estimators in num_estimators_range:
        for learning_rate in learning_rate_range: 
            run_BDT(max_tree_depth, num_estimators, learning_rate)
            
            