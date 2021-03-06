import random

import numpy as np
from pymoo.model.crossover import Crossover
from pymoo.operators.crossover.util import crossover_mask


class LinkageUniformCrossover(Crossover):
    def __init__(self, linkage_prob=0.5, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.linkage_prob = linkage_prob

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape

        # the mask do to the crossover
        M = np.full((n_matings, n_var), False)

        # create for each individual the crossover range
        if random.uniform(0, 1) <= -1:
            M = np.random.random((n_matings, n_var)) < 0.5
        else:
            for i in range(n_matings):
                n_fos = len(kwargs.get("algorithm").model)
                cut = random.uniform(2, np.round(n_fos / 2))
                selected_fos = random.sample(kwargs.get("algorithm").model, int(cut))
                for j in selected_fos:
                    M[i, j] = True

        _X = crossover_mask(X, M)
        return _X
