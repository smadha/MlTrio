'''
Uses flattened features in feature directory and run a SVM on it
'''

from keras.layers import Dense
from keras.models import Sequential
import keras.regularizers as Reg
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
import cPickle as pickle
import numpy as np

import theano
theano.config.openmp = True
OMP_NUM_THREADS=4 

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
    return X_tr



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
        label_new = [0,0]
        label_new[int(label)]=1
        labels_new.append(label_new)
    
    return labels_new

features = pickle.load( open("../feature_engg/feature/all_features.p", "rb") )
labels = transform_label()

print len(features),len(features[0])
print len(labels),len(labels[0])

features = normalize(features)

reg_coeff = 1e-02
momentum = 0.10
eStop = True
sgd_Nesterov = True
sgd_lr = 5e-4
sgd_decay = 5e-05
arch = [len(features[0]),1024,512,2]
batch_size=50000
nb_epoch=50
verbose=True

call_ES = EarlyStopping(monitor='val_acc', patience=6, verbose=1, mode='auto')

# Generate Model
model = genmodel(num_units=arch, reg_coeff=reg_coeff )
# Compile Model
sgd = SGD(lr=sgd_lr, decay=sgd_decay, momentum=momentum, 
    nesterov=sgd_Nesterov)
model.compile(loss='categorical_crossentropy', optimizer=sgd, 
    metrics=['accuracy'])
# Train Model
model.fit(features, labels, nb_epoch=nb_epoch, batch_size=batch_size, 
        verbose=verbose, callbacks=[call_ES], validation_split=0.1, 
        validation_data=None, shuffle=True)

if eStop:
    model.fit(features, labels, nb_epoch=nb_epoch, batch_size=batch_size, 
    verbose=verbose, callbacks=[call_ES], validation_split=0.1, 
    validation_data=None, shuffle=True)
else:
    model.fit(features, labels, nb_epoch=nb_epoch, batch_size=batch_size, 
        verbose=verbose)

# Save model
model.save("model/model_deep.h5")
print("Saved model to disk")
 
 

