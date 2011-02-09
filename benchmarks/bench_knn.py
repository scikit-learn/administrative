"""Various libraries classifying on k-Nearest Neighbors"""

#
#       .. Imports ..
#
import numpy as np
from datetime import datetime
from shogun.Classifier import KNN
from shogun.Features import RealFeatures, Labels
from shogun.Distance import EuclidianDistance
from scikits.learn import neighbors
from mdp.nodes.classifier_nodes import KNNClassifier

#
#       .. Generate dataset ..
#
n_samples, n_dim, n_neighbors = 500, 500, 9
X = 100 * np.random.randn(n_samples, n_dim)
y = np.linspace(1, 10, num=n_samples).astype(np.int32)

print 'Using %s points, %s dims and %s classes' % \
      (n_samples, n_dim, len(np.unique(y)))


def bench_shogun():
#
#       .. Shogun ..
#
    start = datetime.now()
    feat = RealFeatures(X.T)
    distance = EuclidianDistance(feat, feat)
    labels = Labels(y.astype(np.float64))
    knn = KNN(n_neighbors, distance, labels)
    knn.train()
    knn.classify(feat).get_labels()
    return datetime.now() - start


def bench_mdp():
#
#       .. MDP ..
#
    start = datetime.now()
    knn_mdp = KNNClassifier(k=n_neighbors)
    knn_mdp.train(X, y)
    knn_mdp.label(X)
    return datetime.now() - start


def bench_skl():
#
#       .. scikits.learn ..
#
    start = datetime.now()
    clf = neighbors.Neighbors(n_neighbors=n_neighbors)
    clf.fit(X, y)
    clf.predict(X)
    return datetime.now() - start


def bench_mlpy():
#
#       .. MLPy ..
#
    from mlpy import Knn as mlpy_Knn
    start = datetime.now()
    mlpy_clf = mlpy_Knn(n_neighbors)
    mlpy_clf.compute(X, y)
    mlpy_clf.predict(X)
    print 'MLPy timing: ', datetime.now() - start


def bench_pymvpa():
#
#       .. PyMVPA ..
#
    from mvpa.datasets import Dataset
    from mvpa.clfs import knn as mvpa_knn
    start = datetime.now()
    data = Dataset(samples=X, labels=y)
    mvpa_clf = mvpa_knn.kNN()
    mvpa_clf.train(data)
    mvpa_clf.predict(X)
    return datetime.now() - start


if __name__ == '__main__':
    print __doc__
    print 'Shogun: ', bench_shogun()
    print 'MDP: ', bench_mdp()
    print 'scikits.learn: ', bench_skl()
    print 'MLPy: ', bench_mlpy()
    print 'PyMVPA: ', bench_pymvpa()
