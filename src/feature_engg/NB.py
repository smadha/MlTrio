from sklearn import datasets
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import cPickle as pickle

filename = "naive_bayes_model_"

print 'loading user based simple best word feature file'
user_based_train_data = pickle.load(open("./feature/user_based_best_word_features.p", "rb"))

print 'loading user based simple best word label file'
user_based_train_data_lables = pickle.load(open("./feature/user_based_labels.p", "rb"))



def train_NB(train_data, train_label, count):
    print 'training NB'
    
    nb_model = MultinomialNB()
    y_pred_prob = nb_model.fit(train_data, train_label).predict_proba(train_data)
    
    y_pred = nb_model.fit(train_data, train_label).predict(train_data)
    print("No. of mislabeled points out of a total %d points : %d" %(train_data.shape[0], (train_label != y_pred).sum()))
    
    #pickle.dump(y_pred_prob, open("NB_predicted_probabilities.","wb"))
    dump_model(count, nb_model, train_data)
    
def train_user_based_models():
    print 'training user based Naive based models'
    
    user_keys = user_based_train_data.keys()
    count = 0
    for user_key in user_keys:
        print 'user_key', user_key
        print '\n\n'
        print 'user_based_train_data[user_key]', user_based_train_data[user_key]
        print '\n\n'
        print 'user_based_train_data_lables[user_key]', user_based_train_data_lables[user_key]
        print '\n\n'
        train_NB(user_based_train_data[user_key],user_based_train_data_lables[user_key], count)
        count = count + 1
        break
    

def dump_model(file_suffix, nb_model, testdata):
    
    updated_file_name = filename + file_suffix+ ".pkl"
    with open(updated_file_name, 'wb') as fid:
        pickle.dump(nb_model, fid)    

    # load it again
    with open(updated_file_name, 'rb') as fid:
        gnb_loaded = pickle.load(fid)
    
    gnb_loaded.predict(testdata)
    
       
if __name__ == "__main__":
    train_user_based_models()




###### Sample code, on how NBs are trained #############
#     iris = datasets.load_iris()
# 
#     gnb = GaussianNB()
#     print np.shape(iris.data)
#     y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
#     
#     print("No. of mislabeled points out of a total %d points : %d" %(iris.data.shape[0], (iris.target != y_pred).sum()))
#     
#     
#     clf = MultinomialNB()
#     y_pred = clf.fit(iris.data, iris.target).predict_proba(iris.data)
#     print np.unique(iris.target)
#     print y_pred[0:10], iris.target[0:10]