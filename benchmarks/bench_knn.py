"""Various libraries classifying on k-Nearest Neighbors"""

#
#       .. Imports ..
#
import numpy as np
from datetime import datetime
#from shogun import Classifier, Features, Distance
from scikits.learn import neighbors
from mlpy import Knn as mlpy_Knn
from mdp.nodes.classifier_nodes import KNNClassifier
from mvpa.datasets import Dataset
from mvpa.clfs import knn as mvpa_knn

#
#       .. Load dataset ..
#
from load import load_data, bench
print 'Loading data ...'
X, y, T = load_data()
print 'Done, %s samples with %s features loaded into ' \
      'memory' % X.shape
n_neighbors = 9


def bench_shogun():
#
#       .. Shogun ..
#
    start = datetime.now()
    feat = Features.RealFeatures(X.T)
    distance = Distance.EuclidianDistance(feat, feat)
    labels = Features.Labels(y.astype(np.float64))
    test_feat = Features.RealFeatures(T.T)
    knn = Classifier.KNN(n_neighbors, distance, labels)
    knn.train()
    knn.classify(test_feat).get_labels()
    return datetime.now() - start


def bench_mdp():
#
#       .. MDP ..
#
    start = datetime.now()
    knn_mdp = KNNClassifier(k=n_neighbors)
    knn_mdp.train(X, y)
    knn_mdp.label(T)
    return datetime.now() - start


def bench_skl():
#
#       .. scikits.learn ..
#
    start = datetime.now()
    clf = neighbors.NeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X, y)
    clf.predict(T)
    return datetime.now() - start


def bench_mlpy():
#
#       .. MLPy ..
#
    start = datetime.now()
    mlpy_clf = mlpy_Knn(n_neighbors)
    mlpy_clf.compute(X, y)
    mlpy_clf.predict(T)
    print 'MLPy timing: ', datetime.now() - start


def bench_pymvpa():
#
#       .. PyMVPA ..
#
    start = datetime.now()
    data = Dataset(samples=X, labels=y)
    mvpa_clf = mvpa_knn.kNN()
    mvpa_clf.train(data)
    mvpa_clf.predict(T)
    return datetime.now() - start


if __name__ == '__main__':
    print __doc__
    #print 'Shogun: ', bench(bench_shogun)
    print 'MDP: ', bench(bench_mdp)
    print 'scikits.learn: ', bench(bench_skl)
    print 'MLPy: ', bench(bench_mlpy)
    print 'PyMVPA: ', bench(bench_pymvpa)
