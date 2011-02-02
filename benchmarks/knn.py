"""
So here it is: Shogun does not for a distance tree, but always uses
brute force.

So on low dimensions we can have the factor we want, just augment the
number of samples :-)

On the other hand, their brute force is somehow faster than ours ...

"""


from shogun.Classifier import KNN
from shogun.Features import RealFeatures, Labels
from shogun.Distance import EuclidianDistance
from datetime import datetime
import numpy as np

from scikits.learn import neighbors

from mdp.nodes.classifier_nodes import KNNClassifier



n_samples = 500
n_dim = 500
n_neighbors = 9
X = 100 * np.random.randn(n_samples, n_dim)


y = np.linspace(1, 10, num=n_samples).astype(np.int32)

print 'Using %s points, %s dims and %s classes' % (n_samples, n_dim, len(np.unique(y)))

def bench():
    """
    put as function for easy profiling
    """

    start = datetime.now()
    feat = RealFeatures(X.T)
    distance = EuclidianDistance(feat, feat)
    labels = Labels(y.astype(np.float64))

    knn = KNN(n_neighbors, distance, labels)
    knn.train()
    knn.classify(feat).get_labels()
    print 'shogun: ', datetime.now() - start
    del knn

    start2 = datetime.now()
    knn_mdp = KNNClassifier(k=n_neighbors)
    knn_mdp.train(X, y)
    knn_mdp.label(X)
    print 'mdp: ', datetime.now() - start2
    del knn_mdp

    start2 = datetime.now()
    clf = neighbors.Neighbors(n_neighbors=n_neighbors)
    clf.fit(X, y)
    clf.predict(X)
    print 'skl: ', datetime.now() - start2
    del clf

    from mlpy import Knn as mlpy_Knn
    start2 = datetime.now()
    mlpy_clf = mlpy_Knn(n_neighbors)
    mlpy_clf.compute(X, y)
    mlpy_clf.predict(X)
    print 'mlpy: ', datetime.now() - start2

    from mvpa.datasets import Dataset
    from mvpa.clfs import knn as mvpa_knn
    tstart = datetime.now()
    data = Dataset(samples=X, labels=y)
    mvpa_clf = mvpa_knn.kNN()
    mvpa_clf.train(data)
    mvpa_clf.predict(X)
    print
    print 'pymvpa: ', (datetime.now() - tstart)


if __name__ == '__main__':
    bench()
