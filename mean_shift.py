
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs


def get_datas(f):

    fichier = np.load('./npz/' + f)

    x_a = fichier['x']
    y_a = fichier['y']
    z_a = fichier['z']
    t_a = fichier['t']

    # #train = 10000
    # #x_a = x_a[:train]
    # #y_a = y_a[:train]
    # #z_a = z_a[:train]
    # #t_a = t_a[:train]

    X = np.zeros((len(x_a), 2))
    offset = t_a[0]

    for i in range(x_a.shape[0]):

        tx = t_a[i] - offset
        norme = int((x_a[i]**2 + y_a[i]**2 + z_a[i]**2)**0.5)
        # #print(tx, norme)
        X[i][0] = tx
        X[i][1] = norme

    return X

def get_bandwidth(X):
    """
    Doc de estimate_bandwidth
        Estimate the bandwidth to use with the mean-shift algorithm.

        That this function takes time at least quadratic in n_samples. For large
        datasets, it’s wise to set that parameter to a small value.

        Parameters
            X array-like of shape (n_samples, n_features) Input points.

            quantile float, default=0.3 should be between [0, 1] 0.5 means that
            the median of all pairwise distances is used.

            n_samples int, default=None The number of samples to use. If not
            given, all samples are used.

            random_state int, RandomState instance, default=None The generator
            used to randomly select the samples from input points for bandwidth
            estimation. Use an int to make the randomness deterministic.
            See Glossary.

            n_jobs int, default=None The number of parallel jobs to run for
            neighbors search. None means 1 unless in a joblib.parallel_backend
            context. -1 means using all processors. See Glossary for more
            details.

        Returns bandwidth float The bandwidth parameter.
    """

    # The following bandwidth can be automatically detected using
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500, n_jobs=-1)
    print("bandwidth =", bandwidth)
    return bandwidth

def get_cluster(bandwidth, X):
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html

    Doc de MeanShift:

    Mean shift clustering using a flat kernel. Mean shift clustering aims to
    discover “blobs” in a smooth density of samples. It is a centroid-based
    algorithm, which works by updating candidates for centroids to be the mean
    of the points within a given region. These candidates are then filtered in
    a post-processing stage to eliminate near-duplicates to form the
    final set of centroids.

    Seeding is performed using a binning technique for scalability.

    Parameters

        bandwidth float, default=None

            Bandwidth used in the RBF kernel.

            If not given, the bandwidth is estimated using
            sklearn.cluster.estimate_bandwidth;
            see the documentation for that function for hints on scalability
            (see also the Notes, below).

        seeds array-like of shape (n_samples, n_features), default=None

            Seeds used to initialize kernels. If not set, the seeds are
            calculated by clustering.get_bin_seeds with bandwidth as the grid
            size and default values for other parameters.

        bin_seeding bool, default=False

            If true, initial kernel locations are not locations of all points,
            but rather the location of the discretized version of points, where
            points are binned onto a grid whose coarseness corresponds to the
            bandwidth. Setting this option to True will speed up the algorithm
            because fewer seeds will be initialized. The default value is False.
            Ignored if seeds argument is not None.

        min_bin_freq int, default=1

            To speed up the algorithm, accept only those bins with at least
            min_bin_freq points as seeds.

        cluster_all bool, default=True

            If true, then all points are clustered, even those orphans that are
            not within any kernel. Orphans are assigned to the nearest kernel.
            If false, then orphans are given cluster label -1.

        n_jobs int, default=None

            The number of jobs to use for the computation. This works by
            computing each of the n_init runs in parallel.

            None means 1 unless in a joblib.parallel_backend context. -1 means
            using all processors. See Glossary for more details.

        max_iter int, default=300

            Maximum number of iterations, per seed point before the clustering
            operation terminates (for that seed point), if has not converged yet.
    """

    ms = MeanShift( bandwidth=bandwidth,
                    bin_seeding=True,
                    n_jobs=-1,
                    max_iter=500)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("number of estimated clusters : %d" % n_clusters_)
    return labels, cluster_centers, labels_unique, n_clusters_

def plot_cluster(X, labels, cluster_centers, labels_unique, n_clusters_, f):
    plt.figure(1, figsize=(20,10))
    plt.clf()
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]

        plt.plot(   X[my_members, 0],
                    X[my_members, 1],
                    col + '.',
                    alpha=0.051)

        plt.plot(   cluster_center[0],
                    cluster_center[1],
                    'o',
                    markerfacecolor='y',  # col,
                    markeredgecolor='k',
                    markersize=4)

    plt.title(f'Estimated number of clusters: {n_clusters_}\n{f}')
    plt.show()

def main():
    l = os.listdir('./npz')
    band = []
    for f in l:
        fichier = np.load('./npz/' + f)
        X = get_datas(fichier)
        bandwidth = get_bandwidth(X)
        band.append(bandwidth)
        labels, cluster_centers, labels_unique, n_clusters_ = get_cluster(bandwidth, X)
        plot_cluster(X, labels, cluster_centers, labels_unique, n_clusters_, f)
    print(band)

def main1():
    l = os.listdir('./npz')
    for f in l:
        X = get_datas(f)
        for i in range(5):
            bandwidth = 10000 + i*10000
            print(bandwidth)
            labels, cluster_centers, labels_unique, n_clusters_ = get_cluster(bandwidth, X)
            plot_cluster(X, labels, cluster_centers, labels_unique, n_clusters_, f)

# #75292.89657205358,
# #95446.45356769333,
# #95446.45356769333,
# #75258.05384990537,
# #96280.70061220472

if __name__ == '__main__':
    main1()

# bandwidth = 40 000 est bien !
