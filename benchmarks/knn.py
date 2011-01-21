from shogun.Classifier import KNN
from shogun.Features import RealFeatures, Labels
from shogun.Distance import EuclidianDistance
from datetime import datetime
import numpy as np

from scikits.learn import neighbors

n = 1000
k = 9
X , y = np.random.randn(n, 5), np.random.randn(n)


@profile
def bench():
    """
    put as function for easy profiling
    """

    start = datetime.now()
    feat = RealFeatures(X)
    distance = EuclidianDistance(feat, feat)
    labels = Labels(y)

    knn = KNN(k, distance, labels)
    knn_train = knn.train()
    knn.classify(feat).get_labels()
    print knn_train
    print datetime.now() - start


    start2 = datetime.now()
    clf = neighbors.Neighbors(n_neighbors=k)
    clf.fit(X, y)
    clf.predict(X)
    print datetime.now() - start2


if __name__ == '__main__':
    bench()
