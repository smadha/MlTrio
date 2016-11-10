from  simple_expansion.simple_expansion_feature import get_user_feature, users
from dbscan_clustering import run_DBScan
import cPickle as pickle
from clusters import distance_metric

def build_user_data():
    user_features_list = []
    for key in users.keys():
        #print smple_exp.users[key]
        user = users[key]
        user_features_list.append(get_user_feature(user))

    return user_features_list
    
    
def build_similarity_matrix(db):
    
    usersIDVs_clusterId_dict = dict()
    clusterIDVs_userIdList_dict = dict()
    i = 0
    for key in users.keys():
        cluster_id = db.labels_[i]
        usersIDVs_clusterId_dict[key] = cluster_id
        if cluster_id in clusterIDVs_userIdList_dict.keys() and (not clusterIDVs_userIdList_dict[cluster_id] is None):
            clusterIDVs_userIdList_dict[cluster_id].append(key)
        else:
            clusterIDVs_userIdList_dict[cluster_id] = [key]
         
        i += 1   
        
    pickle.dump(clusterIDVs_userIdList_dict, open("clusterIDVs_userIdList_dict.p", "wb"), protocol=2 )
    pickle.dump(usersIDVs_clusterId_dict, open("usersIDVs_clusterId_dict.p", "wb"), protocol=2 )


user_features_list = build_user_data()
#db = run_DBScan(user_features_list)
#build_similarity_matrix(db)
distance_metric.create_metric_matrix(user_features_list, "user_similarity_metric.p", "l1")