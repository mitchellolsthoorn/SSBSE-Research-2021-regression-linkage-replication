import numpy as np
import pandas as pd
import os


def compact_file(data_dir, project, version, file):
    dataset = []
    with open(os.path.join(data_dir, project, version, file), 'r') as reader:
        for line in reader.readlines():
            if line == "\n" or line == "":
                continue
            dataset.append([int(x) for x in line.split(" ") if x != "\n"])

    matrix = np.matrix(dataset)
    matrix_t = matrix.transpose()

    matrix_test = np.transpose(np.matrix(pd.read_csv(os.path.join(data_dir, project, version, file), delim_whitespace=True)))

    column_ids, duplicates = compact_matrix(matrix_t)

    final_array = []
    for i in column_ids:
        squeezed_array = np.squeeze(np.asarray(matrix_t[i]))
        if i in duplicates:
            compressed_column = squeezed_array * (len(duplicates[i]) + 1)
            final_array.append(compressed_column)
        else:
            final_array.append(squeezed_array)

    return np.matrix(final_array).transpose()


def compact_matrix(matrix):
    column_ids = []
    duplicates = {}
    for i in range(matrix.shape[0]):
        duplicate = False
        for j in column_ids:
            if np.array_equal(matrix[i], matrix[j]):
                duplicate = True
                if j in duplicates:
                    duplicates[j].append(i)
                else:
                    duplicates[j] = [i]

        if not duplicate:
            column_ids.append(i)

    return column_ids, duplicates


if __name__ == '__main__':
    data_dir = "../data"
    projects = [
        "bash",
        "flex",
        "grep",
        "sed"
    ]
    versions = {
        "bash": ["v1", "v2", "v3"],
        "flex": ["v1", "v2", "v3"],
        "grep": ["v1", "v2", "v3"],
        "sed": ["v1", "v2", "v3"]
    }

    for project in projects:
        for version in versions[project]:
            print("project: " + project + " - version: " + version)
            res_b = compact_file(data_dir, project, version, "coverage_matrix_b")
            res_f = compact_file(data_dir, project, version, "coverage_matrix_f")
            res_s = compact_file(data_dir, project, version, "coverage_matrix_s")

            np.save(os.path.join(data_dir, project, version, "coverage_matrix_b"), res_b)
            np.save(os.path.join(data_dir, project, version, "coverage_matrix_f"), res_f)
            np.save(os.path.join(data_dir, project, version, "coverage_matrix_s"), res_s)
