from sklearn import metrics
from sklearn.cluster import DBSCAN
import numpy as np
import timeit

def run_DBScan(features):
    
    start_time = timeit.default_timer()
    
    metrix_matrix = metrics.pairwise.pairwise_distances(features, Y=None, metric='euclidean')
    #metrics.pairwise.cosine_similarity(features, dense_output=True)
    print 'metrix_matrix', metrix_matrix
    print 'db_scan start_time::', start_time
    db = DBSCAN(eps=1, min_samples=2, metric='precomputed').fit(metrix_matrix)
    print 'db_scan time taken to cluterise data::', timeit.default_timer() - start_time
    print 'Number of unique cluster lables', np.unique(db.labels_)
    print 'Cluster labels', db.labels_
    print 'Length of cluster labels', len(db.labels_)
    return db
