"""SVM benchmarks"""

import numpy as np
from datetime import datetime
from shogun.Classifier import LibSVM
from shogun.Features import RealFeatures, Labels
from shogun.Kernel import GaussianKernel
from mdp.nodes import LibSVMClassifier
from mvpa.datasets import Dataset
from mvpa.clfs import svm as mvpa_svm
from scikits.learn import svm as skl_svm
from mlpy import LibSvm as mlpy_svm

#
#       .. Load dataset ..
#
from load import load_data, bench
print 'Loading data ...'
X, y, T = load_data()
print 'Done, %s samples with %s features loaded into ' \
      'memory' % X.shape


def bench_shogun():
#
#       .. Shogun ..
#
    start = datetime.now()
    feat = RealFeatures(X.T)
    feat_test = RealFeatures(T.T)
    labels = Labels(y.astype(np.float64))
    kernel = GaussianKernel(feat, feat, 1.)
    shogun_svm = LibSVM(1., kernel, labels)
    shogun_svm.train()
    shogun_svm.classify(feat_test).get_labels()
    return datetime.now() - start


def bench_mlpy():
#
#       .. MLPy ..
#
    start = datetime.now()
    mlpy_clf = mlpy_svm(kernel_type='rbf', C=1.)
    mlpy_clf.learn(X, y.astype(np.float64))
    mlpy_clf.pred(T)
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
#       .. PyBrain ..
#
    raise NotImplementedError


if __name__ == '__main__':
    print __doc__
    print 'Shogun: ', bench(bench_shogun), bench(bench_shogun)
    print 'scikits.learn: ', bench(bench_skl), bench(bench_skl)
    print 'MLPy: ', bench(bench_mlpy), bench(bench_mlpy)
    print 'PyMVPA: ', bench(bench_pymvpa), bench(bench_pymvpa)
    
