import simple_expansion.simple_expansion_feature as simp
import cPickle as pickle
import numpy as np
from collections import Counter
from collections import defaultdict

best_user_words = pickle.load(open("./feature/best_user_words.p", "rb"))
best_ques_words = pickle.load(open("./feature/best_ques_words.p", "rb"))

def get_one_feature(item_set, global_set):
    '''
    item_set - COUNTER of values present in current item
    global_set - COUNTER of values present in whole training set
    return - a feature of length global_set with value set if feature is present in item_set else 0
    '''
    feature_one = []
    for tag in global_set :
        feature_one.append(item_set[tag])
    
    return feature_one


def get_user_feature(user):
    '''

    :param user: user raw data
    :return: user features
    '''
    feature = []
    # # fill features with user vector
    feature.extend(get_one_feature(Counter(user[1].split("/")), simp.user_tags))
    ## can be replaced by word cluster ids
    feature.extend(get_one_feature(Counter(user[2].split("/")), best_user_words))
    ## can be replaced by char cluster ids
    #feature.extend(get_one_feature(Counter(user[3].split("/")), simp.user_char_id))
    
    return feature


def get_ques_feature(question):
    '''

    :param ques: ques raw data
    :return: ques features
    '''

    feature = []
    feature.extend(get_one_feature(Counter(question[1].split("/")), simp.question_tags))
    ## can be replaced by cluster ids
    feature.extend(get_one_feature(Counter(question[2].split("/")), best_ques_words))
    ## can be replaced by cluster ids
    #feature.extend(get_one_feature(Counter(question[3].split("/")), simp.question_char_id))
    ## Fill #upvotes
    feature.append(int(question[4]))
    ## Fill #answers
    feature.append(int(question[5]))
    ## Fill #top quality answers
    feature.append(int(question[6]))
    
    return feature



def get_full_feature(question, user):
    '''

    :param question: question ID
    :param user: user ID
    :return:
    '''
    feature = []
    # # fill features with question vector
    ## Fill tags
    feature.extend(get_ques_feature(question))
    
    #append user feature
    feature.extend(get_user_feature(user))
    
    return feature

def main_fn():
    labels = []
    features = []
    with open(simp.INVITED_INFO_TRAIN) as f:
        training_data = f.readline().strip().split("\t")
        while training_data and len(training_data) == 3 :
            
            labels.append(training_data[2])
            
            question = simp.questions[training_data[0]]
            user = simp.users[training_data[1]]
            
            features.append(get_full_feature(question, user))
            
            if len(features) % 1000 == 0:
                print len(features)
                
            training_data = f.readline().strip().split("\t")
            
            
        
    print "features", len(features)
    print "labels", len(labels)
        
    pickle.dump(features, open("./feature/simple_best_word_features.p", "wb") )
    pickle.dump(labels, open("./feature/labels.p", "wb") )
        
    print "done"  
    
def load_user_based_features():

    user_based_features = defaultdict(list)
    user_based_labels = defaultdict(list)
    features = []
    with open(simp.INVITED_INFO_TRAIN) as f:
        training_data = f.readline().strip().split("\t")
        while training_data and len(training_data) == 3 :
            
            question = simp.questions[training_data[0]]
            user = simp.users[training_data[1]]
            
            features.append(get_full_feature(question, user))
            user_based_features[training_data[1]].append(features)
            user_based_labels[training_data[1]].append(training_data[2])
            if len(features) % 1000 == 0:
                print len(features)
                
            training_data = f.readline().strip().split("\t")
            
    print 'dumping data...'    
    pickle.dump(user_based_features, open("feature/user_based_best_word_features.p", "wb"), protocol=2)
    pickle.dump(user_based_labels, open("feature/user_based_labels.p", "wb"), protocol=2 )
        
    print "done"   
    
if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    #main_fn()
    load_user_based_features() 