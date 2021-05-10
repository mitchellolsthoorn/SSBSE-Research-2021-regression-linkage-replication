import os
import time

import autograd.numpy as np
import pandas as pd
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_crossover, get_mutation, get_sampling, get_performance_indicator
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from FaultDetection import fault_detection
from LinkageUniformCrossover import LinkageUniformCrossover
from NSGA2Linkage import NSGA2Linkage
from Regression import Regression

REPETITIONS = 2


def run():
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
            optimize(data_dir, project, version)


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

    repetitions = []
    reference_front = []
    for repetition in range(0, REPETITIONS):
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
            linkage_frequency=2,
            sampling=get_sampling("bin_random"),
            crossover=LinkageUniformCrossover(0.5),
            mutation=get_mutation("bin_bitflip"),
            eliminate_duplicates=True)

        time1 = time.process_time()
        # Run search
        res1 = minimize(problem,
                        algorithm1,
                        ('n_gen', 200),
                        seed=repetition,
                        verbose=True)
        time1 = time.process_time() - time1
        print(time1)

        time2 = time.process_time()
        # Run search
        res2 = minimize(problem,
                        algorithm2,
                        ('n_gen', 200),
                        seed=repetition,
                        verbose=True)
        time2 = time.process_time() - time2
        print(time2)

        if repetition == 0:
            reference_front = np.row_stack([res1.F, res2.F])
        else:
            reference_front = np.row_stack([reference_front, res1.F, res2.F])

        ns = NonDominatedSorting()
        fronts = ns.do(reference_front)
        reference_front = reference_front[fronts[0], :]

        # problem.visualize(res1.F, res2.F)

        faults_nsga = fault_detection(res1.X, cost_vector, fault_coverage_array)
        faults_ltga = fault_detection(res2.X, cost_vector, fault_coverage_array)
        print("Fault Detection", faults_nsga)
        print("Fault Detection", faults_ltga)

        repetitions.append((project, version, repetition + 1, time1, time2, res1, res2, faults_nsga, faults_ltga))

    res = pd.DataFrame()
    for i in repetitions:
        (project, version, repetition, time1, time2, res1, res2, faults_nsga, faults_ltga) = i
        igd = get_performance_indicator("igd", reference_front, normalize=True)
        igd_nsga = igd.calc(res1.F)
        igd_ltga = igd.calc(res2.F)
        print("igd", igd_nsga)
        print("igd", igd_ltga)

        hv = get_performance_indicator("hv", reference_front.max(axis=0), normalize=True)
        hv_nsga = hv.calc(res1.F)
        hv_ltga = hv.calc(res2.F)
        print("hv", hv_nsga)
        print("hv", hv_ltga)

        res_temp = pd.DataFrame({
            'project': project,
            'version': version,
            'repetition': repetition,
            'nsga_time': time1,
            'ltga_time': time2,
            'nsga_igd': igd_nsga,
            'ltga_igd': igd_ltga,
            'nsga_hv': hv_nsga,
            'ltga_hv': hv_ltga,
            'nsga_faults': faults_nsga,
            'ltga_faults': faults_ltga
        }, index=[0])
        res = res.append(res_temp)

    res.to_csv(os.path.join(data_dir, project, version, "statistics.csv"), index=False)

    # Print results
    # print("Best solution found: %s" % res2.X.astype(int))
    # print("Function value: %s" % res2.F)


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
