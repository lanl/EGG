""" This module contains functions for reading pvals files which are
the output of the GPMMCMC code.
"""

import numpy as np
import sys

def get_file_length(filename):
    """ Returns number of samples in the pvals file. This is the
    number of samples done in the call to the GPMMCMC code.

    Parameters
    ----------
    filename : str
        The path to the pvals file.

    Returns
    -------
    int
        The number of samples in the pvals file.
    """
    with open(filename) as fp:
        for i, l in enumerate(fp):
            pass
    return i + 1

def read_pvals(filename):
    """ Returns a tuple of each ``pvals`` array that is written to the Pvals
    file by the GPMMCMC code. The tuple has the following order:
    ``betaU``, ``lamUz``, ``lamWs``, ``LamWOs``, ``logLik``, ``logPrior``, and
    ``logPost``.

    Parameters
    ----------
    filename : str
        The path to the pvals file.

    Returns
    -------
    betaU : ndarray
    lamUz : ndarray
    lamWs : ndarray
    lamWOs : ndarray
    logLik : ndarray
    logPrior : ndarray
    logPost : ndarray
    """

    # open file and read lines
    with open(filename, 'r') as f:    
        first_line = f.readline()
        split_line = first_line.split()

        # get sizes of dimensions of arrays
        n = int(split_line[0])
        m = int(split_line[1])
        p = int(split_line[2])
        q = int(split_line[3])
        pv = int(split_line[4])
        pu = int(split_line[5])

        # allocate arrays
        #! WARNING: Be careful, this is for n == 0 case which has 7 parameters
        n_trials = int((get_file_length(filename) - 1) / 7)
        betaU = np.zeros((n_trials, pu * (p+q)))
        lamUz = np.zeros((n_trials, pu))
        lamWs = np.zeros((n_trials, pu))
        lamWOs = np.zeros(n_trials)
        logLik = np.zeros(n_trials)
        logPrior = np.zeros(n_trials)
        logPost = np.zeros(n_trials)

        # loop over number of samples    
        for i in range(n_trials):
 
            # read BetaU
            next_line = f.readline()
            split_line = np.asanyarray(next_line.split()).astype(np.float)
            betaU[i,:]=split_line

            # read LamUz
            next_line = f.readline()
            split_line = np.asanyarray(next_line.split()).astype(np.float)
            lamUz[i,:]=split_line

            # read LamWs
            next_line = f.readline()
            split_line = np.asanyarray(next_line.split()).astype(np.float)
            lamWs[i,:]=split_line

            # read LamWOs
            next_line = f.readline()
            split_line = np.asanyarray(next_line.split()).astype(np.float)
            lamWOs[i]=split_line

            # read LogLik
            next_line = f.readline()
            split_line = np.asanyarray(next_line.split()).astype(np.float)
            logLik[i]=split_line

            # read LogPrior
            next_line = f.readline()
            split_line = np.asanyarray(next_line.split()).astype(np.float)
            logPrior[i]=split_line

            # compute LogPost
            logPost[i] = logLik[i]+logPrior[i]

            # skip the blank line
            next_line = f.readline()
            i = i + 1

    return betaU, lamUz, lamWs, lamWOs, logLik, logPrior, logPost

