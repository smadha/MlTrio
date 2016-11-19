import cPickle as pickle 

users_va_te = set([])

## validate_nolabel test_nolabel

def add_all_user(fname):
    global users_va_te
    with open(fname,"r") as f:
        #l = f.readline().strip().split("\t")
        training_data = f.readline().strip().split("\t")
        while training_data and len(training_data) >= 2 :
            user_id = training_data[1]
            if training_data[2] == "1":
                users_va_te.add(user_id.strip())
            training_data = f.readline().strip().split("\t")
            
    print len(users_va_te)

if __name__ == '__main__':

    add_all_user("../bytecup2016data/invited_info_train.txt")
    #add_all_user("../bytecup2016data/validate_nolabel.txt")
    pickle.dump(list(users_va_te) , open("../bytecup2016data/users_va_te.p","wb"), protocol=2)
