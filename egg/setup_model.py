""" This module sets up a model for GPMMCMC from raw data.
"""

# Author: James R. Gattiker, Los Alamos National Laboratory
# Ported to Python by: Christine M. Sweeney, Los Alamos National Laboratory
#
# This file was distributed as part of the GPM/SA software package
# Los Alamos Computer Code release LA-CC-06-079, C-06,114
#
# Copyright 2008.  Los Alamos National Security, LLC. This material 
# was produced under U.S. Government contract DE-AC52-06NA25396 for 
# Los Alamos National Laboratory (LANL), which is operated by Los Alamos 
# National Security, LLC for the U.S. Department of Energy. The U.S. 
# Government has rights to use, reproduce, and distribute this software.  
# NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY 
# WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF 
# THIS SOFTWARE.  If software is modified to produce derivative works, 
# such modified software should be clearly marked, so as not to confuse 
# it with the version available from LANL.
# Additionally, this program is free software; you can redistribute it 
# and/or modify it under the terms of the GNU General Public License as 
# published by the Free Software Foundation; version 2.0 of the License. 
# Accordingly, this program is distributed in the hope that it will be 
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty 
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.

import numpy as np
from egg.classes import *

def setup_model(obsData, simData, verbose=True):
    """ Function for writing model file that is read by the GPMMCMC code.

    A call to this function will write a file to the current working
    directory. Therefore, there should not be concurrent calls to this function
    from the same working directory.

    Parameters
    ----------
    obsData : list
        A list of observed data. At the moment, this must be an empty list.
    simData : dict
        A ``dict`` like returned from ``egg.readdata.readdata``.
    verbose : bool
        If ``True``, then print messages to ``stdout``.

    Returns
    -------
    params : Params
        A ``Params`` instance for the model.
    """

    # check we are being passed what we think we are
    # assume obsData is an empty nparray and  n==0, eta-only model
    # assume scalarOutput is 0
    # assume simData.x not cell
    iscell = not (all(all(isinstance(flt, float) for flt in arr)
                  for arr in simData["x"]))
    if ((obsData == None) or (simData == None) or (obsData != []) or iscell):
        raise TypeError("Only handling case of ObsData empty numpy array, "
                        "simData nonempty and not a cell.")

    # initialize Model instance    
    scOut = 0
    model = Model()
    model.scOut = scOut;

    # do not use observed data
    n = 0
    obsData.append(ObsData())
    obsData[0].x = np.array([])
    obsData[0].Dobs = np.array([])
    obsData[0].yStd = np.array([])

    # set number of input parameters
    p = simData["x"].shape[1]
    q = 0

    # set number of simulations and basis vectors
    m = simData["yStd"].shape[1]
    pu = simData["Ksim"].shape[1]
    pv = 0

    # print model values
    if verbose:
        print("SetupModel: Determined data sizes as follows:")
        if n==0:
          print("SetupModel: This is a simulator (eta) -only model")
          print("SetupModel: m=%3d  (number of simulated data)" % m)
          print("SetupModel: p=%3d  (number of inputs)" % p)
          print("SetupModel: pu=%3d (transformed response dimension)" % pu)
        else:
          print("SetupModel: n=%3d  (number of observed data)" % n)
          print("SetupModel: m=%3d  (number of simulated data)" % m)
          print("SetupModel: p=%3d  (number of parameters known for observations)" % p)
          print("SetupModel: q=%3d  (number of additional simulation inputs (to calibrate))" % q)
          print("SetupModel: pu=%3d (response dimension (transformed))" % pu)
          print("SetupModel: pv=%3d (discrepancy dimension (transformed))" % pv)
        print("")

    # create default lamVzGroup
    lamVzGroup = np.ones((pv, 1), dtype=np.int32)
    lamVzGnum = (np.unique(lamVzGroup)).size

    # create default Data instance
    data = Data()

    # construct the transformed observed data
    if scOut:
        pass

    # else set ridge to be used for stabilization
    # and set empty arrays
    else:
        DKridge = np.eye(pu+pv) * 1e-6
        data.x = np.array([])
        data.v = np.array([])
        data.u = np.array([])

    # iscell will not be true for an eta-only model so leaving this out
    data.zt = simData["x"]
    data.ztSep = np.array([])

    # construct the transformed sim
    data.w = np.linalg.lstsq(simData["Ksim"], simData["yStd"])[0].T
  
    # set estimated calibration variable
    model.theta = 0.5 * np.ones((1, q))

    # set spatial dependence for V discrepancy
    model.betaV = np.ones((p, lamVzGnum)) * 0.1

    # set marginal discrepancy precision
    model.lamVz = np.ones((lamVzGnum, 1)) * 20

    # set PC surface spatial dependence
    model.betaU = np.ones((p + q, pu)) * 0.1

    # set marginal precision
    model.lamUz = np.ones((pu, 1)) * 1

    # set simulator data precision
    model.lamWs = np.ones((pu, 1)) * 1000

    # set initial values and sizes
    model.n = n 
    model.m = m 
    model.p = p 
    model.q = q
    model.pu = pu 
    model.pv = pv
    model.lamVzGnum = lamVzGnum
    model.lamVzGroup = lamVzGroup
    model.w = data.w.T.flatten()
    if scOut:
        pass
    else:
        model.vuw = np.concatenate((data.v.flatten(), data.u.flatten(),
                                    data.w.flatten()))
        model.vu = np.concatenate((data.v.flatten(), data.u.flatten()))

    # compute the PC loadings corrections
    model.LamSim = np.diag(simData["Ksim"].T.dot(simData["Ksim"]))
  
    # initialize the acceptance record field
    model.acc = 1

    # compute LamObs, the u/v spatial correlation
    if scOut:
        pass

    # this part has no effect, for n=0
    # so LO gets set to an empty 2d array of shape 0,0
    else:
        LO = np.zeros((0, 0))

    # compute the Penrose inverse of LO
    model.SigObs = np.linalg.inv(LO) + 1e-8 * np.eye(LO.shape[0])
  
    # set prior distribution types and parameters
    priors = Priors()
    priors.lamVz = Prior()
    priors.lamVz.fname = "gLogGammaPrior"  
    priors.lamVz.params = np.tile(np.array([1, 0.0010]), (lamVzGnum, 1))  
    priors.lamUz = Prior()
    priors.lamUz.fname = "gLogGammaPrior"  
    priors.lamUz.params=np.tile(np.array([5, 5]), (pu, 1)) 
    priors.lamWOs = Prior()
    priors.lamWOs.fname= "gLogGammaPrior"  
    priors.lamWOs.params = np.array([[5, 0.005]])  
    priors.lamWs = Prior()
    priors.lamWs.fname = "gLogGammaPrior"  
    priors.lamWs.params = np.tile(np.array([3, 0.003]), (pu, 1))  
    priors.lamOs = Prior()
    priors.lamOs.fname = "gLogGammaPrior"  
    priors.lamOs.params = np.array([[1, 0.001]])  
    priors.rhoU = Prior()
    priors.rhoU.fname  = "gLogBetaPrior"   
    priors.rhoU.params = np.tile(np.array([1, 0.1]), (pu * (p + q), 1))  
    priors.rhoV = Prior()
    priors.rhoV.fname  = "gLogBetaPrior"   
    priors.rhoV.params = np.tile(np.array([1, 0.1]),(p * lamVzGnum)) 
    priors.theta = Prior()
    priors.theta.fname = "gLogNormalPrior" 
    priors.theta.params = np.tile(np.array([0.5, 10]), (q, 1)) 

    # for lamWOs need K basis correction
    aCorr = 0.5 * (simData["yStd"].shape[0] - pu) * m
    ysimStdHat = simData["Ksim"].dot(data.w.T)
    bCorr = 0.5 * np.sum(np.sum((simData["yStd"] - ysimStdHat)**2))
    priors.lamWOs.params[:, 0] = priors.lamWOs.params[:,0] + aCorr
    priors.lamWOs.params[:, 1] = priors.lamWOs.params[:,1] + bCorr

    # set the initial values of lamOs and lamWOs based on the priors.
    model.lamWOs = max([100.0],
                       priors.lamWOs.params[:, 0] / priors.lamWOs.params[:, 1])
    model.lamOs = max([20.0],
                      priors.lamOs.params[:, 0] / priors.lamOs.params[:, 1])

    # set prior bounds 
    priors.lamVz.bLower = 0
    priors.lamVz.bUpper = float("inf")
    priors.lamUz.bLower = 0.3
    priors.lamUz.bUpper = float("inf")
    priors.lamWs.bLower = 60
    priors.lamWs.bUpper = 1e5
    priors.lamWOs.bLower = 60
    priors.lamWOs.bUpper = 1e5
    priors.lamOs.bLower = 0
    priors.lamOs.bUpper = float("inf")
    priors.betaU = Prior()
    priors.betaV = Prior()
    priors.betaU.bLower = 0
    priors.betaU.bUpper = float("inf")
    priors.betaV.bLower = 0
    priors.betaV.bUpper = float("inf")
    priors.theta.bLower = 0
    priors.theta.bUpper = 1
    priors.theta.constraints = np.array([])

    # set MCMC step interval values
    mcmc = Mcmc()
    mcmc.thetawidth = 0.2 * np.ones((1, np.size(model.theta)))
    mcmc.rhoUwidth = 0.1 * np.ones((1, np.size(model.betaU)))
    mcmc.rhoVwidth = 0.1 * np.ones((1, np.size(model.betaV)))
    mcmc.lamVzwidth = 10 * np.ones((1, np.size(model.lamVz)))
    mcmc.lamUzwidth = 5 * np.ones((1, np.size(model.lamUz)))
    mcmc.lamWswidth = 100 * np.ones((1, np.size(model.lamWs)))
    mcmc.lamWOswidth = 100 * np.ones((1, np.size(model.lamWOs)))
    mcmc.lamOswidth = model.lamOs / 2 * np.ones((model.lamOs.shape))

    # a subset of the params
    mcmc.pvars=["betaU", "lamUz", "lamWs", "lamWOs",
                "logLik", "logPrior", "logPost"]
    mcmc.svars=["betaU", "lamUz", "lamWs", "lamWOs"]

    # set svarSize for betaU, lamUz, lamWs, and lamWOs
    mcmc.svarSize = np.array([[pu * (p + q), pu, pu, 1]])

    # set more params
    mcmc.wvars = ["rhoUwidth", "lamUzwidth", "lamWswidth", "lamWOswidth"]
  
    # create object to return model data
    params = Params()
    params.data = data
    params.model = model
    params.priors = priors
    params.mcmc = mcmc
    params.obsData = obsData
    params.simData = simData
    params.optParms = []
    params.pvals = []

    return params
