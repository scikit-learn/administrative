"""
Shogun is not able to transparently handle the multi-class case ??

WTF ??

"""


from shogun.Classifier import KNN, LibSVM, LibSVMMultiClass
from shogun.Features import RealFeatures, Labels
from shogun.Distance import EuclidianDistance
from shogun.Kernel import GaussianKernel
from datetime import datetime
import numpy as np

from mdp.nodes.classifier_nodes import KNNClassifier

from scikits.learn import neighbors

n_samples = 500
n_dim = 500

X = 100 * np.random.randn(n_samples, n_dim)
y = (10 * np.random.randn(n_samples)).astype(np.int).astype(np.float)
print 'Using %s points, %s dims and %s classes' % (n_samples, n_dim, len(np.unique(y)))


C = 1.

def bench():
    """
    put as function for easy profiling
    """

    start = datetime.now()
    feat = RealFeatures(X.T)
    labels = Labels(y)

    kernel = GaussianKernel(feat, feat, 1.)
    svm = LibSVMMultiClass(C, kernel, labels)
    svm.train()
    print 'shogun: ', datetime.now() - start


    # start2 = datetime.now()
    # knn_mdp = KNNClassifier(k=n_neighbors)
    # knn_mdp.train(X, y)
    # knn_mdp.label(X)
    # print 'mdp: ', datetime.now() - start2

    # start2 = datetime.now()
    # clf = neighbors.Neighbors(n_neighbors=n_neighbors)
    # clf.fit(X, y)
    # clf.predict(X)
    # print 'skl: ', datetime.now() - start2


if __name__ == '__main__':
    bench()
