import os
import time

import autograd.numpy as np
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_crossover, get_mutation, get_sampling, get_performance_indicator
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from FaultDetection import fault_detection
from LinkageUniformCrossover import LinkageUniformCrossover
from NSGA2Linkage import NSGA2Linkage
from Regression import Regression


def run():
    results = {}

    data_dir = "../data"
    projects = ["bash"]
    # projects = [
    #     "bash",
    #     "flex",
    #     "grep",
    #     "sed"
    # ]
    versions = {"bash": ["v1"]}
    # versions = {
    #     "bash": ["v1", "v2", "v3"],
    #     "flex": ["v1", "v2", "v3"],
    #     "grep": ["v1", "v2", "v3"],
    #     "sed": ["v1", "v2", "v3"]
    # }

    for project in projects:
        for version in versions[project]:
            res = optimize(data_dir, project, version)
            results[project + "-" + version] = res


def optimize(data_dir, project, version):
    print("project: " + project + " - version: " + version)

    # Load coverage matrices
    branch_coverage_matrix, function_coverage_matrix, statement_coverage_matrix = load_npy(data_dir, project, version)

    # Load cost vector
    cost_array = []
    with open(os.path.join(data_dir, project, version, "cost_array"), 'r') as reader:
        for line in reader.readlines():
            if line == "\n" or line == "":
                continue
            cost_array.append(int(line))
    cost_vector = np.array(cost_array)

    # Load (next-version) fault coverage
    next_version = "v" + str(int(version.replace("v", "")) + 1)
    fault_coverage_array = []
    count = 0
    with open(os.path.join(data_dir, project, next_version, "fault_matrix"), 'r') as reader:
        for line in reader.readlines():
            if line == "\n" or line == "":
                continue
            fault_coverage_array.append([int(x) for x in line.split(" ") if ((x != "\n") & (x != ""))])
            count += 1
    fault_coverage_array = np.matrix(fault_coverage_array)

    # Define problem
    problem = Regression(len(cost_vector), branch_coverage_matrix, statement_coverage_matrix, cost_vector)

    # Create search algorithm
    algorithm1 = NSGA2(
        pop_size=100,
        sampling=get_sampling("bin_random"),
        crossover=get_crossover("bin_ux"),
        mutation=get_mutation("bin_bitflip"),
        eliminate_duplicates=True)

    # Create search algorithm
    algorithm2 = NSGA2Linkage(
        pop_size=100,
        linkage_frequency=1,
        sampling=get_sampling("bin_random"),
        crossover=LinkageUniformCrossover(0.5),
        mutation=get_mutation("bin_bitflip"),
        eliminate_duplicates=True)

    time1 = time.process_time()
    # Run search
    res1 = minimize(problem,
                    algorithm1,
                    ('n_gen', 200),
                    seed=1,
                    verbose=True)
    time1_res = time.process_time() - time1
    print(time1_res)

    time2 = time.process_time()
    # Run search
    res2 = minimize(problem,
                    algorithm2,
                    ('n_gen', 200),
                    seed=1,
                    verbose=True)
    time2_res = time.process_time() - time2
    print(time2_res)

    problem.visualize(res1.F, res2.F)

    union = np.row_stack([res1.F, res2.F])
    ns = NonDominatedSorting()
    fronts = ns.do(union)
    reference_front = union[fronts[0], :]

    igd = get_performance_indicator("igd", reference_front, normalize=True)
    print("igd", igd.calc(res1.F))
    print("igd", igd.calc(res2.F))

    hv = get_performance_indicator("hv", union.max(axis=0), normalize=True)
    print("hv", hv.calc(res1.F))
    print("hv", hv.calc(res2.F))

    print("Fault Detection", fault_detection(res1.X, cost_vector, fault_coverage_array))
    print("Fault Detection", fault_detection(res2.X, cost_vector, fault_coverage_array))

    # Print results
    # print("Best solution found: %s" % res2.X.astype(int))
    # print("Function value: %s" % res2.F)
    return res1, res2


# Compressed data files
def load_npy(data_dir, project, version):
    # Branches
    branch_coverage_matrix = np.load(os.path.join(data_dir, project, version, "coverage_matrix_b.npy"))

    # Functions
    function_coverage_matrix = np.load(os.path.join(data_dir, project, version, "coverage_matrix_f.npy"))

    # Statements
    statement_coverage_matrix = np.load(os.path.join(data_dir, project, version, "coverage_matrix_s.npy"))

    return branch_coverage_matrix, function_coverage_matrix, statement_coverage_matrix


# Uncompressed data files
def load_ssv(data_dir, project, version):
    # Branches
    branch_coverage_array = []
    with open(os.path.join(data_dir, project, version, "coverage_matrix_b"), 'r') as reader:
        for line in reader.readlines():
            if line == "\n" or line == "":
                continue
            branch_coverage_array.append([int(x) for x in line.split(" ") if ((x != "\n") & (x != ""))])
    branch_coverage_matrix = np.matrix(branch_coverage_array)

    # Functions
    function_coverage_array = []
    with open(os.path.join(data_dir, project, version, "coverage_matrix_f"), 'r') as reader:
        for line in reader.readlines():
            if line == "\n" or line == "":
                continue
            function_coverage_array.append([int(x) for x in line.split(" ") if ((x != "\n") & (x != ""))])
    function_coverage_matrix = np.matrix(function_coverage_array)

    # Statements
    statement_coverage_array = []
    with open(os.path.join(data_dir, project, version, "coverage_matrix_s"), 'r') as reader:
        for line in reader.readlines():
            if line == "\n" or line == "":
                continue
            statement_coverage_array.append([int(x) for x in line.split(" ") if ((x != "\n") & (x != ""))])
    statement_coverage_matrix = np.matrix(statement_coverage_array)

    return branch_coverage_matrix, function_coverage_matrix, statement_coverage_matrix


if __name__ == '__main__':
    run()
