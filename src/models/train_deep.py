'''
Uses flattened features in feature directory and run a SVM on it
'''

from keras.layers import Dense
from keras.models import Sequential
import keras.regularizers as Reg
from keras.optimizers import SGD, RMSprop
from keras.callbacks import EarlyStopping
import cPickle as pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import theano
theano.config.openmp = True
OMP_NUM_THREADS=10 

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



def genmodel(num_units, actfn='relu', reg_coeff=0.0, last_act='softmax'):
    ''' Generate a neural network model of approporiate architecture
    Args:
        num_units: architecture of network in the format [n1, n2, ... , nL]
        actfn: activation function for hidden layers ('relu'/'sigmoid'/'linear'/'softmax')
        reg_coeff: L2-regularization coefficient
        last_act: activation function for final layer ('relu'/'sigmoid'/'linear'/'softmax')
    Output:
        model: Keras sequential model with appropriate fully-connected architecture
    '''

    model = Sequential()
    for i in range(1, len(num_units)):
        if i == 1 and i < len(num_units) - 1:
            model.add(Dense(input_dim=num_units[0], output_dim=num_units[i], activation=actfn, 
                W_regularizer=Reg.l2(l=reg_coeff), init='glorot_normal'))
        elif i == 1 and i == len(num_units) - 1:
            model.add(Dense(input_dim=num_units[0], output_dim=num_units[i], activation=last_act, 
                W_regularizer=Reg.l2(l=reg_coeff), init='glorot_normal'))
        elif i < len(num_units) - 1:
            model.add(Dense(output_dim=num_units[i], activation=actfn, 
                W_regularizer=Reg.l2(l=reg_coeff), init='glorot_normal'))
        elif i == len(num_units) - 1:
            model.add(Dense(output_dim=num_units[i], activation=last_act, 
                W_regularizer=Reg.l2(l=reg_coeff), init='glorot_normal'))
    return model

def transform_label():
    '''
    Returns list of labels as list of [0/1 , 1/0] 
    if label = 1 [0, 1]
    if label = 0 [1, 0]
    '''
    labels = pickle.load( open("../feature_engg/feature/labels.p", "rb") )
    labels_new = []
    for label in labels:
        label_new = [0.0,0.0]
        label_new[int(label)]=1.0
        labels_new.append(label_new)
    
    return labels_new

features = pickle.load( open("../feature_engg/feature/all_features.p", "rb") )
labels = transform_label()

print len(features),len(features[0])
print len(labels),len(labels[0])

features = np.array(features)

col_deleted = np.nonzero((features==0).sum(axis=0) > (len(features)-1000))
print col_deleted
features = np.delete(features, col_deleted, axis=1)

print len(features),len(features[0])
print len(labels),len(labels[0])

features, X_mu, X_sig = normalize(features)

save_res = {"col_deleted":col_deleted,"X_mu":X_mu,"X_sig":X_sig}
with open("model/train_config", 'wb') as pickle_file:
    pickle.dump(save_res, pickle_file, protocol=2)
print "Dumped config"


momentum = 0.99
eStop = True
sgd_Nesterov = True
sgd_lr = 1e-5
batch_size=50000
nb_epoch=100
verbose=True


def run_NN(arch, reg_coeff, sgd_decay, class_weight_0, save=False):
    features_tr, features_te,labels_tr, labels_te = train_test_split(features,labels, train_size = 0.85)
    
    call_ES = EarlyStopping(monitor='val_acc', patience=6, verbose=1, mode='auto')
    
    # Generate Model
    model = genmodel(num_units=arch, reg_coeff=reg_coeff )
    # Compile Model
    sgd = SGD(lr=sgd_lr, decay=sgd_decay, momentum=momentum, 
        nesterov=sgd_Nesterov)
    
    # sgd = RMSprop(lr=sgd_lr, rho=0.9, epsilon=1e-08, decay=sgd_decay)
    
    model.compile(loss='MSE', optimizer=sgd, 
        metrics=['accuracy'])
    # Train Model
    if eStop:
        model.fit(features_tr, labels_tr, nb_epoch=nb_epoch, batch_size=batch_size, 
        verbose=verbose, callbacks=[call_ES], validation_split=0.1, 
        validation_data=None, shuffle=True, class_weight={0:class_weight_0 , 1:1})
    else:
        model.fit(features_tr, labels_tr, nb_epoch=nb_epoch, batch_size=batch_size, 
            verbose=verbose, class_weight={0: class_weight_0 , 1:1})
    
    labels_pred = model.predict(features_te)
    classification_report(labels_te, labels_pred)
    if save:
        # Save model
        model.save("model/model_deep.h5")
        print("Saved model to disk")
    

 
arch_range = [[len(features[0]),1024,2], [len(features[0]),1024,512,2], [len(features[0]),1024,1024,512,2],[len(features[0]),1024,512,256,2]]
reg_coeffs_range = [1e-7, 1e-6, 5e-6, 1e-5]
sgd_decays_range = [1e-5, 5e-5, 1e-4]
class_weight_0_range = [0.5,0.75, 0.9]

#GRID SEARCH ON BEST PARAM
for arch in arch_range:
    for reg_coeff in reg_coeffs_range:
        for sgd_decay in sgd_decays_range:
            for class_weight_0 in class_weight_0_range:
                run_NN(arch, reg_coeff, sgd_decay, class_weight_0)

# arch = [len(features[0]),1024,512,2]
# reg_coeff = 1e-05
# sgd_decay = 1e-05
# class_weight_0 = 0.5
 

