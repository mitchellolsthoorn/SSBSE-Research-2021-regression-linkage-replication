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
                if self.pop[i].data['rank'] in [0, 1]:
                    val.append(self.pop[i].get("X"))
            X = np.array(val).astype(int, copy=False)

            self._train_model(X)

        # do the mating using the current population
        self.off = self.mating.do(self.problem, self.pop, self.n_offsprings, algorithm=self)
        self.off.set("n_gen", self.n_gen)

        # if the mating could not generate any new offspring (duplicate elimination might make that happen)
        if len(self.off) == 0:
            self.termination.force_termination = True
            return

        # if not the desired number of offspring could be created
        elif len(self.off) < self.n_offsprings:
            if self.verbose:
                print("WARNING: Mating could not produce the required number of (unique) offsprings!")

        # evaluate the offspring
        self.evaluator.eval(self.problem, self.off, algorithm=self)

        # merge the offsprings with the current population
        self.pop = Population.merge(self.pop, self.off)

        # the do survival selection
        if self.survival:
            self.pop = self.survival.do(self.problem, self.pop, self.pop_size, algorithm=self,
                                        n_min_infeas_survive=self.min_infeas_pop_size)

    def _train_model(self, X):
        X_t = np.transpose(X)
        Z = hierarchy.linkage(X_t, method='average', metric='hamming')

        cut_off = np.median(np.unique(Z[:, 2]))
        clusters = hierarchy.fcluster(Z, cut_off, criterion='distance')
        fos = []
        for i in np.unique(clusters):
            fos.append(np.where(clusters == i))

        self.model = fos
