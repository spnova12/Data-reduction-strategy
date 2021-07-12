# coding: utf-8
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np

import kmedoids

# 3 points in dataset
data = np.array([
                    [1,1, 1, 1, 1, 1],
                    [1,1, 1, 1, 1, 1],
                    [1,1, 3, 1, 1, 1],
                    [1,1, 3, 3, 3, 1],
                    [1,1, 1, 1, 8, 1],
                    [1,1, 1, 9, 1, 1],
                ])


# distance matrix
# Pairwise distances : http://www.cs.tau.ac.il/~rshamir/algmb/00/scribe00/html/lec08/node17.html
D = pairwise_distances(data, metric='euclidean')

# split into 2 clusters
M_idxs, C = kmedoids.kMedoids(D, 2)

print('medoids:')
for point_idx in M_idxs:
    print(data[point_idx])

print('')
print('clustering result:')
for label in C:
    for point_idx in C[label]:
        print('label {0}:　{1}'.format(label, data[point_idx]))




else:
    print('hi')