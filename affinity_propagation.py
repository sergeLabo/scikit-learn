"""
https://scikit-learn.org/stable/auto_examples/cluster/plot_affinity_propagation.html

=================================================
Demo of affinity propagation clustering algorithm
=================================================

"""

import numpy as np

from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets import make_blobs


# ## Generate sample data
# #centers = [[1, 1], [-1, -1], [1, -1]]
# #X, labels_true = make_blobs(n_samples=300,
                            # #centers=centers,
                            # #cluster_std=0.5,
                            # #random_state=0)
# #print(X)
# #print(labels_true)

# Digits de MNIST
fichier = np.load('12_03_boulanger.npz')

train = 1000
x_a = fichier['x'][:train]
y_a = fichier['y']
z_a = fichier['z']
t_a = fichier['t'][:train]
print("x_a.shape[0] =", x_a.shape[0])

# #X = []
# #for i in range(10000):  # x_a.shape[0]):
    # #X.append((x_a[i], t_a[i]))  # y_a[i], z_a[i],

X = np.stack((x_a, t_a), axis=1)
print("X.shape =", X.shape)

# Compute Affinity Propagation
"""
damping = Damping factor (between 0.5 and 1) is the extent to which the
current value is maintained relative to incoming values (weighted 1 - damping).
This in order to avoid numerical oscillations when updating these values (messages).

max_iterint = Maximum number of iterations.

convergence_iterint = Number of iterations with no change in the number of
estimated clusters that stops the convergence.

preference = Preferences for each point - points with larger values of preferences
are more likely to be chosen as exemplars. The number of exemplars, ie of clusters,
is influenced by the input preferences value. If the preferences are not passed as
arguments, they will be set to the median of the input similarities.
"""

af = AffinityPropagation(   damping=0.7, # default=0.5
                            max_iter=100, # default=200
                            convergence_iter= 30, # default=15
                            preference=None, # default=None
                            verbose="True").fit(X)

cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)

print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels, metric='sqeuclidean'))

# #############################################################################
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

plt.close('all')
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = X[cluster_centers_indices[k]]
    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    for x in X[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
