"""

"""

from datetime import datetime
import numpy as np

# skl import
from scikits.learn import linear_model as skl_lm

# mlpy import
from mlpy import ElasticNet as mlpy_enet


n_samples = 3000
n_dim = 500

X = 100 * np.random.randn(n_samples, n_dim)

# shogun wants its classes to be contiguous integers !!!
# how lame is that ?
y = np.linspace(0, 10, num=n_samples).astype(np.int32).astype(np.float64)

print 'Using %s points, %s dims and %s classes' % (n_samples, n_dim, len(np.unique(y)))


C = 1.

# TODO: set gamma

def bench():
    # as function for easy profiling

    start = datetime.now()

    start = datetime.now()
    skl_clf = skl_lm.ElasticNet(rho=0.5)
    skl_clf.fit(X, y)
    skl_pred = skl_clf.predict(X)
    print
    print 'skl: ', datetime.now() - start


    # TODO: make sure this is what we want.
    start = datetime.now()
    mlpy_clf = mlpy_enet(tau=.5, mu=.5)
    mlpy_clf.learn(X, y)
    mlpy_pred = mlpy_clf.pred(X)
    print
    print 'mlpy: ', datetime.now() - start


    # from mvpa.datasets import Dataset
    # from mvpa.clfs import svm as mvpa_svm
    # tstart = datetime.now()
    # data = Dataset(samples=X, labels=y)
    # clf = mvpa_svm.RbfCSVMC(C=1.)
    # clf.train(data)
    # mvpa_pred = clf.predict(X)
    # print
    # print 'pymvpa: ', (datetime.now() - tstart)
    # print 'Similarity with shogun %s %% ' % 100 * np.mean(mvpa_pred ==  mdp_pred)
    
    
    

if __name__ == '__main__':
    print __doc__
    bench()
