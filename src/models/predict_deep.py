'''
Uses flattened features in feature directory and run a SVM on it
'''

from simple_expansion import simple_expansion_feature as simp
from feature_engg import create_features as eng_feat 
from keras.models import load_model
import numpy as np


loaded_model = load_model("model/model_deep.h5")
print("Loaded model from disk")
 


test_features = []
with open("../../bytecup2016data/validate_nolabel.txt") as f:
    # skip header for prediction
    f.readline()
    test_data = f.readline().strip().split(",")

    while test_data and len(test_data) == 2 :
        question_id = test_data[0]
        user_id = test_data[1]
        
        feature = eng_feat.get_full_feature(question_id, user_id)
        test_features.append(feature)
        
        test_data = f.readline().strip().split(",")
        
        
        
print len(test_features)

col_deleted=np.array([ 13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,        26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,        39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,        52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,        65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,        78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,        91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,       104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117,       118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 131,       132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,       145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157,       158, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171,       172, 173, 174, 175, 176, 177, 178, 179, 181, 182, 183, 184, 185,       186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198,       199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 210, 211, 212,       213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225,       226, 227, 228, 229, 230, 231, 232, 234, 235, 236, 237, 238, 239,       240, 241, 242, 243, 244, 245, 246, 247, 249, 250, 251, 252, 253,       254, 255, 256, 257, 259, 260, 261, 262, 263, 264, 266, 267, 268,       269, 270, 271, 272, 273, 275, 276, 277, 278, 279, 281, 282, 283,       285, 286, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298,       299, 300, 301, 302, 303, 304, 305, 306, 308, 309, 310, 311, 312,       313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 325, 326, 328,       329, 331, 332, 333, 335, 336, 338, 339, 341, 343, 345, 347, 348,       350, 351, 353, 355, 357, 358, 360, 364, 365, 367, 368, 372, 376,       380, 382, 387, 392, 403, 404, 405, 407, 601, 602, 606, 607, 608,       609, 610, 611, 612, 613, 614, 615, 618, 622, 628, 630, 631, 632,       633, 634, 635, 636, 637, 638, 639, 640, 642, 643, 646, 651, 652,       654, 655, 656, 657, 658, 659, 660, 662, 663, 665, 666, 669, 670,       671, 672, 673, 674, 675, 676, 678, 689, 692, 697, 700, 701, 703,       704, 705, 706, 707, 708, 709, 710, 711, 712, 714, 717, 718, 721,       725, 728, 735, 736, 737, 738, 739, 742, 743, 748])
test_features = np.array(test_features)
test_features = np.delete(test_features, col_deleted, axis=1)

print len(test_features)


# predict_proba outputs probability of each class
# [x, y] mean probability of class 0 is x and probability of class 1 is y
test_labels = loaded_model.predict_proba(test_features, verbose=1)

res = open("validate_label.csv", "w")

count = 0
with open("../../bytecup2016data/validate_nolabel.txt") as f:
    # # writing header
    res.write(f.readline())
    test_data = f.readline().strip()
    while test_data :        
        # probability of answering a question is probability in class 1
        prob = test_labels[count][1]
        res.write(test_data + "," + format(prob, '.8f') + "\n")
        count = count + 1
        test_data = f.readline().strip()
    
    
    
    
    
    

