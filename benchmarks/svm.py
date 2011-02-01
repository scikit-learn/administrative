"""
===================================================
Bench multiclass SVMs from shogun and scikits.learn
===================================================

Shogun is not able to transparently handle the multi-class case ??

WTF ??

"""

from shogun.Classifier import LibSVMMultiClass
from shogun.Features import RealFeatures, Labels
from shogun.Kernel import GaussianKernel

from datetime import datetime
import numpy as np

from scikits.learn import svm

n_samples = 5000
n_dim = 50

X = 100 * np.random.randn(n_samples, n_dim)

# shogun wants its classes to be contiguous integers !!!
# how lame is that ?
y = np.linspace(0, 10, num=n_samples).astype(np.int32).astype(np.float64)

print 'Using %s points, %s dims and %s classes' % (n_samples, n_dim, len(np.unique(y)))


C = 1.

def bench():
    # as function for easy profiling

    start = datetime.now()
    feat = RealFeatures(X.T)
    labels = Labels(y)

    kernel = GaussianKernel(feat, feat, 1.)
    shogun_svm = LibSVMMultiClass(C, kernel, labels)
    shogun_svm.train()
    print 'shogun: ', datetime.now() - start


    start2 = datetime.now()
    clf = svm.SVC(kernel='rbf', gamma=1., C=1.)
    clf.fit(X, y)
    print 'skl: ', datetime.now() - start2


    # for some reason can't be imported from top
    # YAY !!!
    from mdp.nodes import LibSVMClassifier


    #$ CRASH !!! $$$$#$#$#$!##Q#$!#$@@@@!@#!@#!
    start2 = datetime.now()
    # no way to set parameters. yeah!!!
    clf = LibSVMClassifier()
    clf.learn(X, y)
    print 'skl: ', datetime.now() - start2


    

if __name__ == '__main__':
    print __doc__
    bench()
