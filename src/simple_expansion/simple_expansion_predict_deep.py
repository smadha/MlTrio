'''
Uses flattened features in feature directory and run a SVM on it
'''

from simple_expansion_feature import questions, users, get_full_feature 
from keras.models import load_model


loaded_model = load_model("model/model_deep.h5")
print("Loaded model from disk")
 


test_features = []
with open("../../bytecup2016data/validate_nolabel.txt") as f:
    # skip header for prediction
    f.readline()
    test_data = f.readline().strip().split(",")

    while test_data and len(test_data) == 2 :
        question = questions[test_data[0]]
        user = users[test_data[1]]
        
        feature = get_full_feature(question, user)
        test_features.append(feature)
        
        test_data = f.readline().strip().split(",")
        
        
print len(test_features)

# predict_proba outputs probability of each class
# [x, y] mean probability of class 0 is x and probability of class 1 is y
test_labels = loaded_model.predict_proba(test_features, verbose=1)

res = open("validate_label.csv", "w")

count = 0
with open("../../bytecup2016data/validate_nolabel.txt") as f:
    # # writing header
    res.write(f.readline())
    test_data = f.readline().strip()
    while test_data :        
        # probability of answering a question is probability in class 1
        prob = test_labels[count][1]
        res.write(test_data + "," + format(prob, '.8f') + "\n")
        count = count + 1
        test_data = f.readline().strip()
    
    
    
    
    
    

