"""K-means clustering"""

import numpy as np
from datetime import datetime
from scikits.learn import cluster as skl_cluster

#
#       .. Generate dataset ..
#
n_samples, n_dim = 500, 500
X = 100 * np.random.randn(n_samples, n_dim)

def bench_skl():
#
#       .. scikits.learn ..
#
    start = datetime.now()
    clf = skl_cluster.KMeans()
    clf.fit(X)
    return datetime.now() - start


if __name__ == '__main__':
    print __doc__
    print 'Using %s points, %s dims' % \
          (n_samples, n_dim)
    print 'scikits.learn: ', bench_skl()

