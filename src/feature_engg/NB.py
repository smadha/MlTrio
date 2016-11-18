from sklearn import datasets
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import cPickle as pickle


print 'loading simple best word feature file'
train_data = pickle.load(open("./feature/simple_best_word_features.p", "rb"))
print 'np.shape(data) ', np.shape(train_data)

train_target_labels = pickle.load(open("./feature/labels.p", "rb"))
print 'np.shape(labels) ', np.shape(train_target_labels)


def train_NB():
    print 'training NB'
    
    clf = MultinomialNB()
    y_pred_prob = clf.fit(train_data, train_target_labels).predict_proba(train_data)
    
    y_pred = clf.fit(train_data, train_target_labels).predict(train_data)
    print("No. of mislabeled points out of a total %d points : %d" %(train_data.shape[0], (train_target_labels != y_pred).sum()))
    
    pickle.dump(y_pred_prob, open("NB_predicted_probabilities.","wb"))
    
    
if __name__ == "__main__":
    train_NB()




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