import cPickle as pickle
import numpy as np

filename = "NBModels/naive_bayes_model_"

print 'loading unique questions....'
unique_ques = pickle.load(open("../../bytecup2016data/ques_va_te.p", "rb"))

print 'loading transformed validation data....'
transformed_validation_feature = pickle.load(open("../feature_engg/feature/validation_best_word_features.p", "rb"))

print 'loading no label validation data....'
old_model_data = np.genfromtxt('model/validate_label_17.csv', delimiter=',')

def predic_prob(data, id):
    updated_file_name = filename + str(id)+ ".pkl"
    with open(updated_file_name, 'rb') as fid:
        gnb_loaded = pickle.load(fid)
    y_pred = gnb_loaded.predict_prob(data)
    return y_pred
    
    
def predict_prob_nolabel_data():
    print 'in'
    count = 0
    with open("../../bytecup2016data/validate_nolabel.txt", 'rb') as f:
        print 'in2'
        validation_data = f.readline().strip().split(",")
        validation_data = f.readline().strip().split(",")

        while validation_data and len(validation_data) == 2 :
            ques_key = validation_data[0]
            if ques_key in unique_ques:
                print predic_prob(transformed_validation_feature[count], ques_key)
            else:
                print old_model_data[count]
            count = count + 1
            if (count%1000) == 0:
                print count
            validation_data = f.readline().strip().split(",")
            
            
predict_prob_nolabel_data()