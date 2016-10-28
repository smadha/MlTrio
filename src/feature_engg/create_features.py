from simple_expansion import simple_expansion_feature as simp
import create_history_features as hist
import create_character_feature as char
from store_character_feature import get_char_pairs
from collections import Counter
import numpy as np
import mltrio_utils
import cPickle as pickle


def get_ques_user_similarity(user_id,ques_id):
    '''
    :param user_id: id of the user
    :param ques_id: id of the question
    :return: L1 and L2 distance between user and question data
    '''
    #TODO: remove the interest tags while comparing
    L1_dist = 0.0
    L2_dist = 0.0
    q_chars = Counter(simp.get_question_char(simp.questions[ques_id]))
    u_chars = Counter(simp.get_user_char(simp.users[user_id]))
    
    q_vec = simp.get_one_feature(q_chars, q_chars+u_chars)
    u_vec = simp.get_one_feature(u_chars, q_chars+u_chars)
    
    L1_dist += mltrio_utils.get_L1_dist(q_vec, u_vec)
    L2_dist += mltrio_utils.get_L2_dist(q_vec, u_vec)

    return L1_dist,L2_dist

def get_full_feature(ques_id, user_id):
    '''
    :param user_id: id of the user_id
    :param ques_id: id of the question_id
    :return: feature combining all hand created features in this package
    '''
    full_feature = []
    full_feature.extend(hist.get_consolidated_feature_train(ques_id, user_id))
    pair_list = get_char_pairs(ques_id, user_id)
    full_feature.extend(char.get_feature(pair_list))
    full_feature.extend(get_ques_user_similarity(user_id,ques_id))
    
    return full_feature
    
if __name__ == '__main__':
    features = []
    labels = []
    with open(simp.INVITED_INFO_TRAIN) as f:
        training_data = f.readline().strip().split("\t")
        while training_data and len(training_data) == 3 :
            label = training_data[2]
            question_id = training_data[0]
            user_id = training_data[1]
            
            features.append(get_full_feature(question_id, user_id))
            labels.append(label)
#             print len(features[0]),features[0]
            training_data = f.readline().strip().split("\t")
            
            
    
    pickle.dump(features, open("./feature/all_features.p", "wb"), protocol=2 )
    pickle.dump(labels, open("./feature/labels.p", "wb"), protocol=2 )
    