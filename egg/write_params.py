""" This module contains functions writing a ``params.h`` header file
that will be compiled with the ``emu.c`` C code.
"""

import numpy as np

def write_c_var(var, var_name, dtype, filename, append):
    """ Function for writing an array to C header file in proper format.

    Parameters
    ----------
    var : ndarray
        A ``numpy.ndarray`` that contains the parameter values.
    var_name : str
        A string of the parameters name.
    dtype : str
        A string such as ``static int`` that describes the C type.
    filename : str
        Path to the C header file that is being written.
    append : bool
        If True, then open C header file in append mode. Otherwise, open
        in write mode.
    """

    # open file
    if append:
        fid = open(filename, "a")
    else:
        fid = open(filename, "w")
    
    # figure out the declaration
    decl = dtype + " " + var_name
    if var.size > 1:
        for lng in var.shape:
            if lng > 1:
                decl = decl + "[" + str(lng) + "]";            
    decl = decl + " = "
    if var.size > 1:
        decl = decl + "{\n"
    
    # write the declaration
    fid.write(decl)
    
    # write the actual data
    var.tofile(fid, sep=",", format="%s")

    # write end of declaration and close file
    if var.size > 1:
        fid.write("};\n\n")
    else:
        fid.write(";\n\n")
    fid.close()

def write_params(params, pvec, beta, lamws, lamuz, n_samples,
                 filename="params.h"):
    """ This function writes a C header file that is compiled with the
    ``emu.c`` C code to emulate a particular problem.

    A call to this function will write a file to the current working
    directory. Therefore, there should not be concurrent calls to this function
    from the same working directory.

    Parameters
    ----------
    params : Params
        A ``Params`` instance.
    pvec : ndarray
        An array of indices that is used to calculate the mean ``beta``,
        ``lamws``, and ``lamUz`` parameters to write to file.
    beta : ndarray
        An array for ``beta`` parameter.
    lamws : ndarray
        An array for ``lamws`` parameter.
    lamuz : ndarray
        An array for ``Lamuz`` parameter.
    n_samples : int
        Number of samples to generate while running compiled ``emu.exe``.
    filename : str
        Path of C header file to write.
    """

    # save to C header file the sizes of dimensions in model
    write_c_var(np.array(params.model.m), "m", "static int", filename, False)
    write_c_var(np.array(params.model.p), "p", "static int", filename, True)
    write_c_var(np.array(params.model.pu), "peta", "static int",
                filename, True)
    write_c_var(np.array(params.simData["yStd"].shape[0]),
                "neta", "static int", filename, True)

    # save the size of random sample to generate 
    write_c_var(np.array(n_samples), "samp", "static int", filename, True)

    # save the x, xmin, xrange, and xmax of design data
    write_c_var(params.simData["x"], "x", "static double",
                filename, True)
    write_c_var(params.simData["orig"]["xmin"],
                "xmin", "static double", filename, True)
    write_c_var(params.simData["orig"]["xrange"],
                "xrange", "static double", filename, True)
    write_c_var(params.simData["orig"]["xmin"] +
                    params.simData["orig"]["xrange"],
                "xmax", "static double", filename, True)

    # save Ksim
    write_c_var(params.simData["Ksim"], "K", "static double", filename, True)

    # save w
    write_c_var(params.data.w.transpose(), "w", "static double",
                filename, True)

    # save the mean beta parameters
    #! FIXME: some day we want to save a sample
    beta_mean = np.mean(beta[pvec,:],0)
    write_c_var(beta_mean.reshape(params.model.pu, params.model.p),
                "beta", "static double", filename, True)

    # save the mean lamWs and lamUz
    lamws_mean = np.mean(lamws[pvec,:],0)
    write_c_var(lamws_mean, "lamws", "static double", filename, True)
    lamuz_mean = np.mean(lamuz[pvec,:],0)
    write_c_var(lamuz_mean, "lamz", "static double", filename, True)

    # ymean, ysd
    write_c_var(params.simData["orig"]["ymean"], "mean", "static double",
                filename, True)
    write_c_var(params.simData["orig"]["ysd"], "sd", "static double",
                filename, True)

    # add in the cholSigmaSim declaration
    fid = open(filename, "a")
    decl = "static double cholSigmaSim[" + str(params.model.pu) + "]["  + \
               str(params.model.m) + "][" + str(params.model.m) +"];\n\n"
    fid.write(decl)
    fid.close()

    # add in the KrigBasis declaration
    fid = open(filename, "a")
    decl = "static double KrigBasis[" + str(params.model.pu) + "][" + \
               str(params.model.m) +"];\n\n"
    fid.write(decl)
    fid.close()

    # add in the VarBasis declaration
    fid = open(filename, "a")
    decl = "static double VarBasis[" + str(params.model.pu) + "][" + \
               str(params.model.m) +"];"
    fid.write(decl)
    fid.close()
