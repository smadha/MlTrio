import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np

def plot_histogram(x, var_name, bins=10):
    '''
    plots histogram of x array, 10 bins equal size
    '''
    _, hist_data, _ = plt.hist(x, bins=bins)
    print type(hist_data), hist_data
    plt.plot(x=hist_data)
    plt.savefig(var_name, linewidth=0)
    plt.close()

def plot_best_histogram(x):
    hist, bins = np.histogram(x, bins="auto")
    print len(hist), len(bins) 
    
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    
    plt.savefig("best-freq_0-freq_1", linewidth=0)
    plt.close()
    
    print np.digitize([-1,-0.5,0,0.5,1], bins)

pair_score = pickle.load(open("./feature/distinguish_char.p", "r"))

# for bins in range(0,151,10):
#     if bins == 0:
#         bins = 10
#     plot_histogram(pair_score.values(), str(bins) + "-freq_0-freq_1", bins)

plot_best_histogram(pair_score.values())


