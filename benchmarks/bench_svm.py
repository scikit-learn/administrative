
import numpy as np
from datetime import datetime
from shogun.Classifier import LibSVMMultiClass
from shogun.Features import RealFeatures, Labels
from shogun.Kernel import GaussianKernel
from mdp.nodes import LibSVMClassifier
from mvpa.datasets import Dataset
from mvpa.clfs import svm as mvpa_svm
from scikits.learn import svm as skl_svm
from mlpy import LibSvm as mlpy_svm


#
#       .. Generate dataset ..
#
n_samples, n_dim = 500, 500
X = 100 * np.random.randn(n_samples, n_dim)
y = np.linspace(0, 10, num=n_samples).astype(np.int32).astype(np.float64)

print 'Using %s points, %s dims and %s classes' % \
      (n_samples, n_dim, len(np.unique(y)))

C = 1.


def bench_shogun():
#
#       .. Shogun ..
#
    start = datetime.now()
    feat = RealFeatures(X.T)
    labels = Labels(y.astype(np.float64))
    kernel = GaussianKernel(feat, feat, 1.)
    shogun_svm = LibSVMMultiClass(C, kernel, labels)
    shogun_svm.train()
    return datetime.now() - start


def bench_mlpy():
#
#       .. MLPy ..
#
    start = datetime.now()
    mlpy_clf = mlpy_svm(kernel_type='rbf', C=1.)
    mlpy_clf.learn(X, y)
    mlpy_clf.pred(X)
    return datetime.now() - start


def bench_skl():
#
#       .. scikits.learn ..
#
    start = datetime.now()
    clf = skl_svm.SVC(kernel='rbf', C=1.)
    clf.fit(X, y)
    clf.predict(X)
    return datetime.now() - start


def bench_pymvpa():
#
#       .. PyMVPA ..
#
    tstart = datetime.now()
    data = Dataset(samples=X, labels=y)
    clf = mvpa_svm.RbfCSVMC(C=1.)
    clf.train(data)
    clf.predict(X)
    return datetime.now() - tstart


def bench_pybrain():
#
#       .. PyMVPA ..
#
    raise NotImplementedError


if __name__ == '__main__':
    print __doc__
    print 'Shogun: ', bench_shogun()
    print 'scikits.learn: ', bench_skl()
    print 'MLPy: ', bench_mlpy()
