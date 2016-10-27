'''
compute Cartesian product of user and question characters 
'''

from collections import Counter
from simple_expansion import simple_expansion_feature as simp
import cPickle as pickle

# Each class has it's own Counter to keep track of pair frequency in that class
pairs_class_i = [Counter([]), Counter([])]
list_all_pairs = []
# score for all pairs 
pair_score = {}

def add_pair(pair, class_i):
    '''
    Adds a pair to Counter of class_i tuples
    class_i = 0/1 based on class id
    pair = One pair of character observed in question, user
    '''
    class_i = int(class_i)
    pairs_class_i[class_i].update([pair])
    
with open(simp.INVITED_INFO_TRAIN) as f:
    training_data = f.readline().strip().split("\t")
    count = 0
    while training_data and len(training_data) == 3 :
        label = training_data[2]
        question = training_data[0]
        user = training_data[1]
        
        user_chars = simp.get_user_char(simp.users[user])
        ques_chars = simp.get_question_char(simp.questions[question])
        
        for user_char in user_chars:
            for ques_char in ques_chars:
                add_pair((user_char, ques_char), label)
            
        training_data = f.readline().strip().split("\t")
        count += 1
        if(count % 1000) == 0:
            print count, "processed"
            break



#list_all_pairs is all possible list of pairs found training data
list_all_pairs = set(pairs_class_i[0].keys() + pairs_class_i[1].keys())

print "Unique char pairs in class0", len(pairs_class_i[0])
print "Unique char pairs in class1", len(pairs_class_i[1])
print "Unique char pairs in training data", len(list_all_pairs)


for pair in list_all_pairs:
    freq_0 = pairs_class_i[0][pair]
    freq_1 = pairs_class_i[1][pair]
    score = 1.0 * (freq_0 - freq_1) / sum([freq_0,freq_1])
    pair_score[pair] = score
    
        
print len(pair_score)
print len(list_all_pairs)

pickle.dump( pair_score, open("./feature/distinguish_char.p", "wb") )

