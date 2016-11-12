import cPickle as pickle
import numpy as np
from  simple_expansion.simple_expansion_feature import users
# from clusters import user_similarity_clusters
 
no_of_splits= 8

fname= "user_hamming_similarity_metric"
user_hamming_distance_metric  = pickle.load(open(fname+"_0.p", "rb") )
for i in range(1, no_of_splits):
    user_hamming_distance_metric  = np.vstack((user_hamming_distance_metric, pickle.load(open(fname+"_"+str(i)+".p", "rb") )))
print 'size of l2 distance metric', np.shape(user_hamming_distance_metric) 

# 
# fname= "user_l2_similarity_metric"
user_l2_distance_metric = []
#user_l2_distance_metric  = pickle.load(open(fname+"_1.p", "rb") )
# for i in range(2, no_of_splits+1):
#     user_l2_distance_metric  = np.vstack((user_l2_distance_metric, pickle.load(open(fname+"_"+str(i)+".p", "rb") )))
# print 'size of l2 distance metric', np.shape(user_l2_distance_metric) 


def get_distance_metric_for_user(U1,U2, metric):
    
    '''
        Returns a scalar value : distance measure between two users
    '''
    u1_index = users.keys().index(U1)
    u2_index = users.keys().index(U2)
    
    if metric== "l2":
        return user_l2_distance_metric[u1_index][u2_index]
    else:
        return user_hamming_distance_metric[u1_index][u2_index]

    
if __name__ == '__main__':

    #print get_distance_metric_for_user("f37635bcd54355ced64ff62b113bc692", "4588a1df2461674252ff01c63b59171a", "l2")
    print get_distance_metric_for_user("f37635bcd54355ced64ff62b113bc692", "4588a1df2461674252ff01c63b59171a", "hamming")
    
    