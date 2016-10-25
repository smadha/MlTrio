from sklearn import metrics
from sklearn.cluster import DBSCAN
import numpy as np
import timeit

def run_DBScan(features):
    
    start_time = timeit.default_timer()
    
    metrix_matrix = metrics.pairwise.cosine_similarity(features, dense_output=True)
    print 'db_scan start_time::', start_time
    db = DBSCAN(eps=0.7, min_samples=2).fit(metrix_matrix)
    print 'db_scan time taken to cluterise data::', timeit.default_timer() - start_time
    print db
    print 'cluster lables', np.unique(db.labels_)
    print 'length of cluster labels', len(db.labels_)

#run_DBScan(features)