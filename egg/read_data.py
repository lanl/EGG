""" This module contains functions for reading data.
"""

import numpy as np

def read_data(design_file, out_file, pu, min_vals=None, max_vals=None):
    """ This function returns a ``dict`` of the input parameters and
    original output data from files, as well as a standardized version
    of the output data.

    Parameters
    ----------
    design_file : str
        Path to design file.
    out_file : str
        Path to data output file.
    pu : int
        Size of basis.
    min_vals : list
        List of minimum values for parameters.
    max_vals : list
        List of maximum values for parameters.

    Returns
    -------
    sim_data : dict
        A ``dict`` with input data and basis.
    """

    # read and standardize the design
    design = np.loadtxt(design_file, skiprows=1, delimiter=",")
    m, p = design.shape
    if min_vals is None:
        x_min = design.min(0)
    else:
        x_min = min_vals
    if max_vals is None:
        x_max = design.max(0)
    else:
        x_max = max_vals
    x_range = x_max - x_min
    design = (design - x_min) / x_range

    # read the output
    ysim = np.loadtxt(out_file, skiprows=1, delimiter=",")
    ysim = np.transpose(ysim)

    # standardize the output
    neta = ysim.shape[0]
    ysimmean = ysim.mean(1)
    ysimStd = ysim - ysimmean.reshape(neta, 1)
    ysimsd = ysimStd.std()
    ysimStd = ysimStd / ysimsd

    # construct the basis
    u, s, _ = np.linalg.svd(ysimStd, 0)
    ksim = np.dot(u[:, 0:pu], np.diag(s[0:pu])) / np.sqrt(m)

    # print explained variance from chosen basis
    s_sq = s * s
    per_var = 100 * sum(s_sq[0:pu]) / sum(s_sq)
    print("These basis vectors explain {}% of the variance".format(per_var))

    # make a dictionary of data
    orig_data = {
        "y" : ysim,
        "ymean" : ysimmean,
        "ysd" : ysimsd,
        "xmin" : x_min,
        "xrange" : x_range,
    }
    sim_data = {
        "x" : design,
        "yStd" : ysimStd,
        "Ksim" : ksim,
        "orig" : orig_data,
    }

    return sim_data
