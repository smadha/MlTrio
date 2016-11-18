import cPickle as pickle 

users_va_te = set([])

## validate_nolabel test_nolabel

def add_all_user(fname):
    with open(fname,"r") as f:
        f.readline().strip().split(",")
        training_data = f.readline().strip().split(",")
        while training_data and len(training_data) >= 2 :
            user_id = training_data[1]
            users_va_te.add(user_id)
            training_data = f.readline().strip().split(",")
            
    

if __name__ == '__main__':
    add_all_user("../bytecup2016data/test_nolabel.txt")
    add_all_user("../bytecup2016data/validate_nolabel.txt")
    pickle.dump(users_va_te, open("../bytecup2016data/users_va_te.p","wb"), protocol=2)