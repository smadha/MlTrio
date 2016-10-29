import cPickle as pickle

cluster_user_dict = pickle.load(open("../clusters/cluster_user_dict.p", "rb") )
users_cluster_dict = pickle.load(open("../clusters/users_cluster_dict.p", "rb") )

def get_total_cluster():
    '''
    Returns total number of user clusters found
    '''
    return len(cluster_user_dict)

def get_user_cluster_vector(user_id):
    '''
    Returns a 0/1 array set for cluster user_id belong to
    '''
    vect = [0] * get_total_cluster()
    cluster_id = users_cluster_dict[user_id] + 1
    vect[cluster_id] = 1
    
    return vect
    
if __name__ == '__main__':
#     print len(cluster_user_dict), cluster_user_dict[-1]
#     print len(users_cluster_dict), users_cluster_dict[users_cluster_dict.keys()[0]]
#     print set(users_cluster_dict.values())
    print get_user_cluster_vector("1ff1de774005f8da13f42943881c655f")
    
    