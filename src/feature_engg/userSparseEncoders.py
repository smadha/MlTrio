from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

import numpy as np
import cPickle as pickle

users_new = []
encoding_dim = 10


def build_user_data():
    users_id_dict = {}
    from simple_expansion.simple_expansion_feature import users, get_user_feature
    index = 0
    with open("../../bytecup2016data/user_info.txt") as f:
        user_data = f.readline().strip().split("\t")
        while user_data and len(user_data) == 4 :
            user_id = user_data[0]
            users_id_dict[user_id] = index
            user_feature = get_user_feature(user_data)
            users_new.append(user_feature)
            user_data = f.readline().strip().split("\t")
            index += 1
     
    print 'np.shape(users_new)::', np.shape(users_new)
    pickle.dump(users_new, open("user_complete_features.p", "wb"))   
    pickle.dump(users_id_dict, open("users_ids_dict.p", "wb"))
    print users_new[0]
build_user_data() 


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

user_data = pickle.load( open("user_complete_features.p", "rb") )
print np.shape(user_data)

[user_data, data_mu, data_sig] = normalize(user_data)

user_data_train = user_data[0:1000]
user_data_test = user_data[1000:1500]

def train_encoder_decoder():
    
    num_of_features = np.shape(user_data_train)[1]
    
    print 'number of features', num_of_features
    input_features = Input(shape=(num_of_features,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_features)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(num_of_features, activation='linear')(encoded)
    
    # this model maps an input to its reconstruction
    autoencoder = Model(input=input_features, output=decoded)
    encoder = Model(input=input_features, output=encoded)
    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
    autoencoder.compile(optimizer='adadelta', loss='mse')
    
    autoencoder.fit(user_data_train, user_data_train,
                    nb_epoch=100,
                    batch_size=1000,
                    shuffle=True,
                    validation_data=(user_data_test, user_data_test))
    
    autoencoder.save("user_encoder_model_deep.h5")
    
    print "user_data_test first training example:", user_data_test[0]
    compressed_features = encoder.predict(user_data_test)
    
    pickle.dump(compressed_features, open("encodedFeatures/compressed_user_features.p", "wb"))
    
    print compressed_features[0]
    decoded_imgs = decoder.predict(compressed_features)
    print '\n', decoded_imgs[0, 0:30]
    
train_encoder_decoder()
