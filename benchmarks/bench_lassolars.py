"""bench different LARS implementations"""

from datetime import datetime
import numpy as np

from scikits.learn import linear_model
from mlpy import Lasso as mlpy_lasso



#
#       .. Load dataset ..
#
from load import load_data, bench
print 'Loading data ...'
X, y, T = load_data()
print 'Done, %s samples with %s features loaded into ' \
      'memory' % X.shape
n_neighbors = 9


def bench_skl():
    start = datetime.now()
#    skl_clf = linear_model.LassoLARS(alpha=0.)
#    skl_clf.fit(X, y, normalize=False)
 #   skl_clf.predict(X)
    linear_model.lars_path(X, y)
    print
    print 'skl: ', datetime.now() - start
    

def bench_mlpy():


#    our lasso does a couple more of steps, different parameters
    start = datetime.now()
    mlpy_clf = mlpy_lasso(m=X.shape[1])
    mlpy_clf.learn(X, y)
    mlpy_clf.pred(X)
    print
    print 'mlpy: ', datetime.now() - start


    from mvpa.datasets import Dataset
    from mvpa.clfs import lars as mvpa_lars
    tstart = datetime.now()
    data = Dataset(samples=X, labels=y)
    mvpa_clf = mvpa_lars.LARS()
    mvpa_clf.train(data)

    # BROKEN
#    mvpa_pred = mvpa_clf.predict(X)
    print
    print 'pymvpa: ', (datetime.now() - tstart)
    # print 'Similarity with shogun %s %% ' % 100 * np.mean(mvpa_pred ==  mdp_pred)
    
    


if __name__ == '__main__':
    print __doc__
    bench_skl()
    bench_mlpy()
