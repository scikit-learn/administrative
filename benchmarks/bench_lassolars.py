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
from mdp.nodes import LibSVMClassifier

from datetime import datetime
import numpy as np

from scikits.learn import linear_model

from mlpy import Lasso as mlpy_lasso

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

    print 'Shogun: NO LARS'
    # start = datetime.now()
    # feat = RealFeatures(X.T)
    # labels = Labels(y)
    # kernel = GaussianKernel(feat, feat, 1.)
    # shogun_svm = LibSVMMultiClass(C, kernel, labels)
    # shogun_svm.train()
    # shogun_pred = shogun_svm.classify(feat).get_labels()
    # print 'shogun: ', datetime.now() - start

    start = datetime.now()
    skl_clf = linear_model.LassoLARS(alpha=0.)
    skl_clf.fit(X, y)
    skl_pred = skl_clf.predict(X)
    print
    print 'skl: ', datetime.now() - start
    # print 'Similarity with SKL %s %% ' % 100 * np.mean(shogun_pred == skl_pred)

    # start = datetime.now()
    # mdp_clf = LibSVMClassifier(kernel='rbf', params={'C':1})
    # mdp_clf.train(X, y)
    # mdp_pred = mdp_clf.label(X)
    # print
    # print 'mdp: ', datetime.now() - start
    # print 'Similarity with shogun %s %% ' % 100 * np.mean(shogun_pred ==  mdp_pred)


    # our lasso does a couple more of steps, different parameters
    # start = datetime.now()
    # mlpy_clf = mlpy_lasso(m=n_dim)
    # mlpy_clf.learn(X, y)
    # mlpy_pred = mlpy_clf.pred(X)
    # print
    # print 'mlpy: ', datetime.now() - start
    # print 'Similarity with skl %s %% ' % 100 * np.mean(mlpy_pred ==  mdp_pred)


    from mvpa.datasets import Dataset
    from mvpa.clfs import lars as mvpa_lars
    tstart = datetime.now()
    data = Dataset(samples=X, labels=y)
    clf = mvpa_lars.LARS()
    clf.train(data)
    mvpa_pred = clf.predict(X)
    print
    print 'pymvpa: ', (datetime.now() - tstart)
    # print 'Similarity with shogun %s %% ' % 100 * np.mean(mvpa_pred ==  mdp_pred)
    
    
    

if __name__ == '__main__':
    print __doc__
    bench()
