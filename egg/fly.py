""" This module contains functions for calling the emulation code.
"""

import ast
import egg
import numpy as np
import os
import shutil
import subprocess
import sys
from egg.read_data import read_data
from egg.read_pvals import read_pvals
from egg.setup_model import setup_model
from egg.write_params import write_params

def fly(input_file, rand_samples=None, burn_in=None, params_file="params.h",
        pvals_file="emuPvals.txt", hatch_file="hatch_inputs.txt",
        tmp_dir="tmp"):
    """ This function compiles and calls the emulation code.

    A call to this function will write several files to the current working
    directory. Therefore, there should not be concurrent calls to this function
    from the same working directory. The emulation code will be compiled in
    ``tmp_dir`` and the executable copied back to the current working
    directory. In addition, the output of the emulation code is written to
    several ``.dat`` files.

    Parameters
    ----------
    input_file : str
        Path to file that contains input parameters to be emulated.
    rand_samples : int
        Number of random samples to use from traning set.
    burn_in : int
        Number of samples to include in burn in. The value should be between
        the range of 0 and 1.
    params_file : str
        Path to C header file with parameters.
    pvals_file : str
        Path to Pvals file from GPMMCMC code.
    hatch_file : str
        Path to file from calling ``egg.hatch.hatch``.
    tmp_dir : str
        Path to temporary directory to compile C emulator code.

    Returns
    -------
    y : ndarray
        The emulator output for the set of input parameters given in
        ``input_file``.
    """

    # find number of lines in input file
    with open(input_file) as f:
        lines = f.readlines()

    # read input file 
    x = np.loadtxt(input_file)    
       
    # if number of input is 1 then put that into a list and write
    ipset = []
    if len(lines) == 1:
        ipset.append(x)                   
        np.savetxt("xstar.dat", ipset, delimiter=" ")
    else:
        np.savetxt("xstar.dat", x, delimiter=" ")

    # read pvals file
    beta_u, lam_uz, lam_ws, lam_wos, _, _, _ = read_pvals(pvals_file)

    # recreate Params object from call to hatch
    # read in variables used during hatch
    with open(hatch_file, "r") as text_file:
        hatch_inputs = text_file.read()
    hatch_dict = ast.literal_eval(hatch_inputs)

    # get our params variable set up
    sim_data = read_data(hatch_dict["designFile"], hatch_dict["simOutFile"],
                         hatch_dict["numBases"], hatch_dict["minvals"],
                         hatch_dict["maxvals"]) 
    params = setup_model([], sim_data)

    # get the length of the samples Markov Chain
    # (i.e., total MCMC samples of each param)
    n_samples = hatch_dict["numofSamples"] 

    # initialize defaults for burn in and number of random samples
    default_burn_in = 0.25
    default_rand_samples = 0

    # use burn in defaults
    if burn_in is None:
        burn_in = default_burn_in
        print("Default initial burn-in period: "
              "first %d%% of samples" % (100 * burn_in))

    # use user-provided burn in value
    elif burn_in < 0 or burn_in >= 1:
        burn_in = default_burn_in
        print("Error: burn_in must be in [0,1)")
        print("Using default initial burn-in period: "
              "first %d%% of samples" % (100 * burn_in))

    # use default of roughly 5% of training samples
    if rand_samples is None:
        rand_samples = default_rand_samples
        print("By default, %d random samples will "
              "be generated." % rand_samples)

    # use user-provided number of training samples
    elif (not isinstance(rand_samples, int)) or rand_samples < 0:
        print("Error: rand_samples must be a positive integer")
        rand_samples = default_rand_samples
        print("Resorting to default number of random samples "
              "to generate: %d" % rand_samples)

    # write the parameter values
    # exclude the first burn_in% of MCMC draws
    pvec = np.arange(int(n_samples * burn_in), n_samples, 1)
    write_params(params, pvec, beta_u, lam_ws, lam_uz, rand_samples)

    # change into temporary directory for building C code
    # copy parameters file into temporary directory as well
    orig_dir = os.getcwd()
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    shutil.copy2(params_file, tmp_dir + "/params.h")
    shutil.copymode(params_file, tmp_dir + "/params.h")
    shutil.copystat(params_file, tmp_dir + "/params.h")
    os.chdir(tmp_dir)

    # compile the emu
    data_dir = "/".join([egg.__path__[0], "emu"])
    subprocess.check_call(["cp", data_dir + "/makefile",
                           data_dir + "/emu.c", "."])
    subprocess.check_call("make")

    # copy executable to original directory and remove temporary directory
    os.chdir(orig_dir)
    shutil.copy2(tmp_dir + "/emu.exe", orig_dir + "/emu.exe")
    shutil.copymode(tmp_dir + "/emu.exe", orig_dir + "/emu.exe")
    shutil.copystat(tmp_dir + "/emu.exe", orig_dir + "/emu.exe")
    shutil.rmtree(tmp_dir)

    # get the answers for the test design
    os.system("./emu.exe")

    # read output
    y = np.loadtxt("ystar.dat")

    return y
