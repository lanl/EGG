""" This module contains functions for calling the GPMMCMC code.
"""

import egg
import numpy as np
import os
import shutil
import stat
import subprocess
import sys
from egg.read_data import read_data
from egg.read_pvals import read_pvals
from egg.setup_model import setup_model
from egg.write_model import write_model

def hatch(design_file, sim_file, num_samples, gpu_on,
          basis_size=15, min_vals=None, max_vals=None, tmp_dir="tmp"):
    """ This function builds and runs the GPMMCMC code.

    A call to this function will write several files to the current working
    directory. Therefore, there should not be concurrent calls to this function
    from the same working directory. The GPMMCMC code will be compiled in
    ``tmp_dir`` and the executable copied back to the current working
    directory. In addition, the output of the GPMMCMC code is written to
    ``emuPvals.txt``, the model used by the GPMMCMC code is written to
    ``emuModel.txt``, and the inputs to this function call is written to
    in ``hatch_inputs.txt``.

    Parameters
    ----------
    design_file : str
        Path to file that contains parameter values.
    sim_file : str
        Path to file that contains simulation outputs.
    num_samples : int
        Number of samples to run with GPMMCMC.
    gpu_on : bool
        If set to ``True`` use GPU version, else use serial version,
    basis_size : int
        Number of basis vectors to use.
    min_vals : list
        Save minimum values for parameters in ``hatch_inputs.txt``.
    max_vals : list
        Save maximum values for parameters in ``hatch_inputs.txt``.
    tmp_dir : str
        Path to temporary directory to compile C++ code. This directory will
        be removed.

    Returns
    -------
    params : Params
        Returns the ``Params`` instance for the model.
    """

    # read simulation data and setup model
    simData = read_data(design_file, sim_file, basis_size, min_vals, max_vals) 
    params = setup_model([], simData)

    # write intermediate file that stores model data
    # that the GPMMCMC code reads
    model_file = "emuModel.txt"    
    write_model(params, model_file)

    # path of file that GPMCMC code writes its output to
    pval_file = "emuPvals.txt"

    # change into temporary directory for building C++ code
    orig_dir = os.getcwd()
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    os.chdir(tmp_dir)

    # build the GPMMCMC code
    if gpu_on:
        mod_path = "/".join([egg.__path__[0], "gpmmcmc", "gpu"])
        exe_name = "gpmmcmc_gpu"
    else:
        mod_path = "/".join([egg.__path__[0], "gpmmcmc", "serial"])
        exe_name = "gpmmcmc"
    os.system("cp {}/* .".format(mod_path))
    os.system("make")

    # copy executable to original directory and remove temporary directory
    os.chdir(orig_dir)
    shutil.copy2(tmp_dir + "/" + exe_name, orig_dir + "/" + exe_name)
    shutil.copymode(tmp_dir + "/" + exe_name, orig_dir + "/" + exe_name)
    shutil.copystat(tmp_dir + "/" + exe_name, orig_dir + "/" + exe_name)
    shutil.rmtree(tmp_dir)

    # run the GPMCMC code
    cmd = [orig_dir + "/" + exe_name, num_samples, model_file, pval_file]
    cmd = cmd + ["1"] if gpu_on else cmd
    cmd = " ".join(map(str, cmd))
    os.system(cmd)

    # save input settings so we can use them later
    hatchDict = {
        "designFile": design_file,
        "simOutFile": sim_file,
        "numBases": basis_size,
        "minvals": min_vals,
        "maxvals": max_vals,
        "numofSamples": num_samples,
    }
    with open("hatch_inputs.txt", "w") as fp:
        fp.write(repr(hatchDict))

    return params
