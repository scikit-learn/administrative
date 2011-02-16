
import numpy as np

def load_data():
    f = open('data/gisette_train.data')
    X = np.fromfile(f, dtype=np.float64, sep=' ')
    f.close()
    X = X.reshape(-1, 5000)

    f = open('data/gisette_train.labels')
    y = np.fromfile(f, dtype=np.float64, sep=' ')
    f.close()

    return  X, y
    
    
def bench(func):
    try:
        a = func()
    except Exception as detail:
        print 'Something went wrong in function %s: '
        print detail
    return a
