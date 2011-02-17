
import numpy as np

def load_data():
    f = open('data/madelon_train.data')
    X = np.fromfile(f, dtype=np.float64, sep=' ')
    f.close()
    X = X.reshape(-1, 500)

    f = open('data/madelon_train.labels')
    y = np.fromfile(f, dtype=np.int32, sep=' ')
    f.close()

    f = open('data/madelon_test.data')
    T = np.fromfile(f, dtype=np.float64, sep=' ')
    f.close()
    T = T.reshape(-1, 500)

    return  X, y, T
    
    
def bench(func):
    try:
        a = func()
    except Exception as detail:
        print '%s error in function %s: ' % (repr(detail), func)
        return
    return a
