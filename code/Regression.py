import autograd.numpy as np

from pymoo.model.problem import Problem
from pymoo.visualization.scatter import Scatter


class Regression(Problem):
    def __init__(self,
                 number_of_test_cases,
                 branch_coverage_matrix,
                 statement_coverage_matrix,
                 cost_vector,
                 ):
        super().__init__(n_var=number_of_test_cases, n_obj=3, n_constr=0, xl=0, xu=1)

        # fitness functions:
        # 1. minimize overall test execution duration
        # 2. maximize overall test suite branch coverage
        # 3. maximize overall test suite statement coverage

        self.number_of_test_cases = number_of_test_cases
        self.branch_coverage_matrix = branch_coverage_matrix
        self.statement_coverage_matrix = statement_coverage_matrix
        self.cost_vector = cost_vector
        self.max_branch_compression = np.transpose(np.max(branch_coverage_matrix, axis=0))
        self.max_statement_compression = np.transpose(np.max(statement_coverage_matrix, axis=0))
        self.model = []

    def _evaluate(self, x, out, *args, **kwargs):
        ff1 = np.dot(self.cost_vector, np.transpose(x))
        ff2 = -np.dot(np.clip(np.dot(x, self.branch_coverage_matrix), 0, 1), self.max_branch_compression)
        ff3 = -np.dot(np.clip(np.dot(x, self.statement_coverage_matrix), 0, 1), self.max_statement_compression)

        out["F"] = np.column_stack([ff1, ff2, ff3])

    @staticmethod
    def visualize(F1, F2):
        F3 = np.copy(F1)
        F3[:, 0] = F1[:, 0]
        F3[:, 1] = -F1[:, 1]
        F3[:, 2] = -F1[:, 2]
        F4 = np.copy(F2)
        F4[:, 0] = F2[:, 0]
        F4[:, 1] = -F2[:, 1]
        F4[:, 2] = -F2[:, 2]
        plot = Scatter(angle=(30, -45))
        plot.add(F3, color="blue")
        plot.add(F4, color="red")
        plot.show()
