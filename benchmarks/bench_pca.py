"""K-means clustering"""

import numpy as np
from datetime import datetime
from scikits.learn import pca as skl_pca
from pybrain.auxiliary import pca as pybrain_pca
import mdp
from mvpa.mappers.pca import PCAMapper as MVPA_PCA
from mvpa.datasets import Dataset

#
#       .. Generate dataset ..
#
n_samples, n_dim = 500, 500
n_components = 9
X = 100 * np.random.randn(n_samples, n_dim)



def bench_skl():
#
#       .. scikits.learn ..
#
    start = datetime.now()
    clf = skl_pca.RandomizedPCA(n_components=n_components)
    clf.fit(X)
    return datetime.now() - start


def bench_pybrain():
#
#       .. pybrain ..
#
    start = datetime.now()
    pybrain_pca.pca(X, n_components)
    return datetime.now() - start



def bench_mdp():
#
#       .. MDP ..
#
    start = datetime.now()
    mdp.pca(X, output_dim=n_components)
    return datetime.now() - start

def bench_mvpa():
#
#       .. PyMVPA ..
#
    start = datetime.now()
    clf = MVPA_PCA()
    data = Dataset(samples=X, labels=0)
    clf.train(data)
    print 'Warning, PyMVPA does not accept keyword to set number ' \
          'of components'
    return datetime.now() - start
    


if __name__ == '__main__':
    print __doc__
    print 'Using %s points, %s dims' % \
          (n_samples, n_dim)
    print 'scikits.learn: ', bench_skl()
    print 'pybrain: ', bench_pybrain()
    print 'MDP: ', bench_mdp()
    print 'PyMVPA: ', bench_mvpa()
