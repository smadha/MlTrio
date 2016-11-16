from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.optimizers import SGD, RMSprop
from keras.callbacks import EarlyStopping

import numpy as np
import cPickle as pickle

ques_new = []


## Hyper parameters for encoding question features:
encoding_dim = [10000]
regularisation_val = [10e-5]
loss_func = ["kld"]
batch_size = [10000]


def build_ques_data():
    from simple_expansion.simple_expansion_feature import get_ques_feature
     
    ques_id_dict = {}
    index =0       
    with open("../../bytecup2016data/question_info.txt") as f:
        question_data = f.readline().strip().split("\t")
        while question_data and len(question_data) == 7 :
            #ques_id = question_data[0]
            ques_id_dict[question_data[0]] = index
            ques_feature = get_ques_feature(question_data)
            ques_new.append(ques_feature)
            question_data = f.readline().strip().split("\t")
            index += 1
    print 'np.shape(ques_new)::', np.shape(ques_new)
    pickle.dump(ques_new, open("ques_complete_features.p", "wb"))   
    pickle.dump(ques_id_dict, open("question_ids_dict.p", "wb"))
    
    
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

#build_ques_data() 

ques_data = pickle.load( open("ques_complete_features.p", "rb") )
print np.shape(ques_data)
[ques_data, data_mu, data_sig] = normalize(ques_data)

print 'ques_data[0]', ques_data[0]

ques_data_train = ques_data[0:7000]
ques_data_test = ques_data[7000:8095]
 
def train_ques_encoder_decoder():
     
    num_of_features = np.shape(ques_data_train)[1]
    #call_ES = EarlyStopping(monitor='val_acc', patience=6, verbose=1, mode='auto')
    print 'number of features', num_of_features
    input_features = Input(shape=(num_of_features,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim[0], activation='relu',activity_regularizer=regularizers.activity_l1(regularisation_val[0]))(input_features)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(num_of_features, activation='linear')(encoded)
     
    # this model maps an input to its reconstruction
    autoencoder = Model(input=input_features, output=decoded)
    encoder = Model(input=input_features, output=encoded)
    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim[0],))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
    #ksgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    autoencoder.compile(optimizer='adadelta', loss=loss_func[0])
     
    autoencoder.fit(ques_data_train, ques_data_train,
                    nb_epoch=100,
                    batch_size=batch_size[0],
                    shuffle=True,
                    validation_data=(ques_data_test, ques_data_test))
    
    model_file_name = "ques_encoder_model_deep_"+loss_func[0]+".h5"
    autoencoder.save(model_file_name) 
    
    print "ques_data_test first training example:", ques_data_test[0]
    compresses_features = encoder.predict(ques_data)
    pickle.dump(compresses_features, open("encodedFeatures/compressed_ques_features_mse.p", "wb"))
    print compresses_features[0]
     
train_ques_encoder_decoder()
