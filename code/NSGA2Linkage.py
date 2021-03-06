import random

from autograd import numpy as np
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.population import Population
from scipy.cluster import hierarchy


class NSGA2Linkage(NSGA2):
    def __init__(self, linkage_frequency=10, **kwargs):
        super().__init__(**kwargs)
        self.linkage_frequency = linkage_frequency
        self.model = []

    def _next(self):
        if (self.n_gen - 1) % self.linkage_frequency == 0:
            val = []
            for i in range(len(self.pop)):
                if self.pop[i].data['rank'] in [0]:
                    val.append(self.pop[i].get("X"))

            X = np.array(val) #.astype(int, copy=True)
            # if len(val) == 100:
            #   X = X[:50, :]

            self._train_model(X)

        super()._next()

    def _train_model(self, X):
        X_t = np.transpose(X)
        Z = hierarchy.linkage(X_t, method='average', metric='hamming')

        # we look at clusters with non-zero distance
        non_zeros = Z[Z[:, 2] > 0, 2]
        cut_off = np.median(non_zeros)

        clusters = hierarchy.fcluster(Z, cut_off, criterion='distance')
        fos = []
        for i in np.unique(clusters):
            fos.append(np.where(clusters == i))

        self.model = fos
