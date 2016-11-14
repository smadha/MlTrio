import numpy as np
import cPickle as pickle
from simple_expansion.simple_expansion_feature import questions


"""

    For this file to execute properly, copy all the question distance metric files (2 files in total)
    -ques_l2_similarity_metric.p,  ques_hamming_similarity_metric.p to the folder var defined below
    and call get_distance_metric_for_ques() method as directed in the init method below.
    
"""

folder = "dist_data"

def get_distance_metric_for_ques(Q1,Q2, metric):
    '''
        metric : expected values : l2 / hamming
        Returns a scalar value : distance measure between two users
    '''

    if metric == 'l2':
        ques_distance_metric  = pickle.load(open(folder+"ques_l2_similarity_metric.p", "rb") )
    else:
        ques_distance_metric  = pickle.load(open(folder+'ques_hamming_similarity_metric.p', "rb"))

    q1_index = questions.keys().index(Q1)
    
    q2_index = questions.keys().index(Q2)
    print q1_index, q2_index
    print np.shape(ques_distance_metric)
    return ques_distance_metric[q1_index][q2_index]


if __name__ == '__main__':
#     print len(cluster_user_dict), cluster_user_dict[-1]
#     print len(users_cluster_dict), users_cluster_dict[users_cluster_dict.keys()[0]]
#     print set(users_cluster_dict.values())
    print get_distance_metric_for_ques("367edcb36424493a7cf80f70903a64cd", "58695dc8b407850d8a028cec784f535c", 'l2')
    print get_distance_metric_for_ques("367edcb36424493a7cf80f70903a64cd", "58695dc8b407850d8a028cec784f535c", 'hamming')