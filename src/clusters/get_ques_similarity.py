import numpy as np
from  simple_expansion.simple_expansion_feature import questions

def get_distance_metric_for_ques(Q1,Q2):
    '''
        Returns a scalar value : distance measure between two users
    '''
    
    ques_distance_metric  = np.genfromtxt(open('ques_features_metric.p', 'r'))
    print 'getting question key indexes'
    q1_index = questions.keys().index(Q1)
    print 'getting question key keys'
    q2_index = questions.keys().index(Q2)
    return ques_distance_metric[q1_index][q2_index]


if __name__ == '__main__':
#     print len(cluster_user_dict), cluster_user_dict[-1]
#     print len(users_cluster_dict), users_cluster_dict[users_cluster_dict.keys()[0]]
#     print set(users_cluster_dict.values())
    print get_distance_metric_for_ques("367edcb36424493a7cf80f70903a64cd", "58695dc8b407850d8a028cec784f535c")