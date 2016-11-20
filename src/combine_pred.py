'''
Created on 19-Nov-2016

@author: madhav
'''
import numpy as np

files = ["../bytecup2016data/test_label_item_based.csv","../bytecup2016data/test_label_graphlab_rating.csv"]

files_open = [open(fi,"r") for fi in files]

with open("../bytecup2016data/test_label_avg2.csv", "w") as fo:
    fo.write(files_open[0].readline())
    [f.readline() for f in files_open[1:]]
    lineArr = [f.readline().strip().split(",") for f in files_open]
    while len(lineArr[0]) == 3:
        
        all_pred = np.array([ float(l[2]) for l in lineArr] ) 
#         print lineArr[0][0] , lineArr[0][1], all_pred, sum(all_pred)/len(all_pred)
        fo.write(lineArr[0][0] + "," + lineArr[0][1] + "," + str(np.average(all_pred)) )
        fo.write("\n")
        
        lineArr = [f.readline().strip().split(",") for f in files_open]
        
    