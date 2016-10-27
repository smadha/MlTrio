import dbscan_clustering
from  src.simple_expansion.simple_expansion_feature import get_user_feature, users
import cPickle as pickle

def build_user_data():
    user_features_list = []
    for key in users.keys():
        #print smple_exp.users[key]
        user = users[key]
        user_features_list.append(get_user_feature(user))

    return user_features_list

def build_user_cluster_dict(db):
    
    users_cluster_dict = dict()
    cluster_user_dict = dict()
    i = 0
    for key in users.keys():
        cluster_id = db.labels_[i]
        users_cluster_dict[key] = cluster_id
        if cluster_id in cluster_user_dict.keys() and (not cluster_user_dict[cluster_id] is None):
            cluster_user_dict[cluster_id].append(key)
        else:
            cluster_user_dict[cluster_id] = [key]
         
        i += 1   
        
    pickle.dump(cluster_user_dict, open("cluster_user_dict.p", "wb") )
    pickle.dump(users_cluster_dict, open("users_cluster_dict.p", "wb") )


user_features_list = build_user_data()
db = dbscan_clustering.run_DBScan(user_features_list)
build_user_cluster_dict(db)
