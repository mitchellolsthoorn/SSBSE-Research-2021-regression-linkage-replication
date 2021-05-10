import autograd.numpy as np
import numpy


def fault_detection(pareto_set, cost_vector, fault_matrix):
    cost_data = np.dot(cost_vector, np.transpose(pareto_set))
    fault_data = np.sum(np.clip(np.dot(pareto_set, fault_matrix), 0, 1), axis=1)

    # sort points in ascending order of cost
    results = np.column_stack([cost_data, fault_data])
    results = np.row_stack([[0, 0], results])  # add first point
    results = results[results[:, 0].argsort()]  # sorting

    # compute the area under the curve (AUC)
    AUC = numpy.trapz(y=results[:, 1], x=results[:, 0])

    # Normalize the AUC
    AUC /= np.max(cost_data)
    AUC /= np.shape(fault_matrix)[1]

    return AUC
