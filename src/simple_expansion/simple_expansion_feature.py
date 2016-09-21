'''
Dumps flattened features in feature directory in below format
Q_1F_1    Q_1F_2    Q_1F_3    U_1F_1    U_1F_2    U_1F_3    U_1F_4    0
Q_2F_1    Q_2F_2    Q_2F_3    U_2F_1    U_2F_2    U_2F_3    U_2F_4    1
..
..
Q_nF_1    Q_nF_2    Q_nF_3    U_nF_1    U_nF_2    U_nF_3    U_nF_4    0

'''
from collections import Counter
import cPickle as pickle

  
# user id to user map
users = {}
user_word_id = Counter({})
user_char_id = Counter({})
user_tags = Counter({})
# question id to question map
questions = {}
question_word_id = Counter({})
question_char_id = Counter({})
question_tags = Counter({})

with open("../../bytecup2016data/user_info.txt") as f:
    user_data = f.readline().strip().split("\t")
    while user_data and len(user_data) == 4 :
        users[user_data[0]] = user_data
        user_tags.update(user_data[1].split("/"))
        user_word_id.update(user_data[2].split("/"))
        user_char_id.update(user_data[3].split("/"))
        user_data = f.readline().strip().split("\t")

print "users", len(users)

with open("../../bytecup2016data/question_info.txt") as f:
    question_data = f.readline().strip().split("\t")
    while question_data and len(question_data) == 7 :
        questions[question_data[0]] = question_data
        question_tags.update(question_data[1].split("/"))
        question_word_id.update(question_data[2].split("/"))
        question_char_id.update(question_data[3].split("/"))
        question_data = f.readline().strip().split("\t")
     
       
print "questions", len(questions)
def get_one_feature(item_set, global_set):
    '''
    item_set - COUNTER of values present in current item
    global_set - COUNTER of values present in whole training set
    return - a feature of length global_set with value set if feature is present in item_set else 0
    '''
    feature_one = []
    for tag in global_set.keys() :
        feature_one.append(item_set[tag])
    
    return feature_one

def get_full_feature(question, user):
    feature = []
    # # fill features with question vector
    ## Fill tags
    feature.extend(get_one_feature(Counter(question[1].split("/")), question_tags))
    ## can be replaced by cluster ids 
    feature.extend(get_one_feature(Counter(question[2].split("/")), question_word_id))
    ## can be replaced by cluster ids
    feature.extend(get_one_feature(Counter(question[3].split("/")), question_char_id))
    ## Fill #upvotes
    feature.append(question[4])
    ## Fill #answers
    feature.append(question[5])
    ## Fill #top quality answers
    feature.append(question[6])
    
    # # fill features with user vector
    feature.extend(get_one_feature(Counter(user[1].split("/")), user_tags))
    ## can be replaced by cluster ids
    feature.extend(get_one_feature(Counter(user[2].split("/")), user_word_id))
    ## can be replaced by cluster ids
    feature.extend(get_one_feature(Counter(user[3].split("/")), user_char_id))
    
    return feature
  
def main_fn():
    labels = []
    features = []
    with open("../../bytecup2016data/invited_info_train.txt") as f:
        training_data = f.readline().strip().split("\t")
        while training_data and len(training_data) == 3 :
            
            labels.append(training_data[2])
            
            question = questions[training_data[0]]
            user = users[training_data[1]]
            
            feature = get_full_feature(question, user)
            
            training_data = f.readline().strip().split("\t")
            features.append(feature)
            
        
    print "features", len(features)
    print "labels", len(labels)
        
    pickle.dump(features, open("./feature/simple_features.p", "wb") )
    pickle.dump(labels, open("./feature/labels.p", "wb") )
        
    print "done"    
    
if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main_fn() 