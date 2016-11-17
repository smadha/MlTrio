'''
Trains a boosting tree
'''
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import cPickle as pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from models.train_bdt import run_BDT


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

# features = np.random.normal(size=(2294,354))
# labels = [0]*2000 + [1]*(2294-2000)

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


run_BDT(RandomForestClassifier, 2, 50, 1, save=True, test=False)
            
            