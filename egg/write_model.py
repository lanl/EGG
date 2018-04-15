""" This module contains functions for writing a model file that is read
by the compiled GPMMCMC code.
"""

def write_model(params, filename):
    """ This function writes out the data for a GPM/SA model, to be read by
    the C version of the likelihood and sampling code.

    A call to this function will write a file to the current working
    directory. Therefore, there should not be concurrent calls to this function
    from the same working directory.

    Parameters
    ----------
    params : Params
        A ``Params`` instance.
    filename : str
        Path to write file.
    """

    # get number of observations
    n = params.model.n

    # get number of simulations
    m = params.model.m

    # get number of observation indpendent variables
    p = params.model.p

    # get number of additional simulator independent variables
    q = params.model.q

    # get size of the discrepancy (delta) basis dependent variable
    pv = params.model.pv

    # get size fo the response (eta) basis transformed depdendent variable
    pu = params.model.pu

    # open file to write
    with open(filename, "w") as f:

        # write sizes of dimensions of arrays
        f.write("%d %d %d %d %d %d\n" % (n, m, p, q, pv, pu))
        
        # write the observation independent variables
        for item in params.data.x.T:
            f.write("%f" % item)
        f.write("\n")
        
        # write the simulator independent variables
        for itemX in params.data.zt.T:
            for item in itemX:
                f.write("%f " % item)
        f.write("\n")
        
        # write the observation discrepancy and eta response
        for item in params.model.vu:
            f.write("%f" % item)
        f.write("\n")
        
        # write the simulator eta response
        for item in params.model.w.T:            
            f.write("%f " % item)
        f.write("\n")
        
        # initial values for lamOs model parameter(s)
        for item in params.model.lamOs:  
            f.write("%f " % item)
        f.write("\n")
        
        # initial values for lamWOs model parameter(s)
        for item in params.model.lamWOs:  
            f.write("%f " % item)
        f.write("\n")
        
        # initial values for theta model parameter(s)
        #for item in params.model.theta:  
        #    f.write("%f " % item)
        f.write("\n")
        
        # initial values for betaV model parameter(s)
        #for item in params.model.betaV:  
        #    f.write("%f " % item)
        f.write("\n")
        
        # initial values for lamVz model parameter(s)
        #for item in params.model.lamVz:  
        #    f.write("%f " % item)
        f.write("\n")
        
        # initial values for betaU model parameter(s)
        for itemX in params.model.betaU:
            for item in itemX:
                f.write("%f " % item)
        f.write("\n")
        
        # initial values for lamUz model parameter(s)
        for itemX in params.model.lamUz:
            for item in itemX:
                f.write("%f " % item)
        f.write("\n")
        
        # initial values for lamWs model parameter(s)
        for itemX in params.model.lamWs:
            for item in itemX:
                f.write("%f " % item)
        f.write("\n")
        
        # initial values for LamSim model parameter
        for item in params.model.LamSim:
            f.write("%f " % item)        
        f.write("\n")
        
        # initial values for SigObs model parameter
        f.write("\n")
        
        # write prior and MCMC parameters
        #if n > 0 :
        #    vars_a = ["theta" "rhoV"  "rhoU" 
        #              "lamVz" "lamUz" "lamWs" "lamWOs" "lamOs"]
        #    vars_b = ["theta" "betaV" "betaU"
        #              "lamVz" "lamUz" "lamWs" "lamWOs" "lamOs"]
        #else:
        #    vars_a = ["rhoU"  "lamUz"  "lamWs"  "lamWOs"]
        #    vars_b = ["betaU" "lamUz"  "lamWs"  "lamWOs"]

        # write MCMC step, lower bound, upper bound, and prior params
        f.write("%f " % params.mcmc.rhoUwidth[0, 0])
        f.write("%f " % params.priors.betaU.bLower)
        f.write("%f " % params.priors.betaU.bUpper)
        #f.write("%f " % params.priors.rhoU.params[0, :])
        for item in params.priors.rhoU.params[0, :]:  
            f.write("%f " % item)
        f.write("\n")

        # write MCMC step, lower bound, upper bound, and prior params
        f.write("%f " % params.mcmc.lamUzwidth[0, 0])
        f.write("%f " % params.priors.lamUz.bLower)
        f.write("%f " % params.priors.lamUz.bUpper)
        #f.write("%f " % params.priors.rhoU.params[0, :])
        for item in params.priors.lamUz.params[0, :]:  
            f.write("%f " % item)
        f.write("\n")

        # write MCMC step, lower bound, upper bound, and prior params
        f.write("%f " % params.mcmc.lamWswidth[0, 0])
        f.write("%f " % params.priors.lamWs.bLower)
        f.write("%f " % params.priors.lamWs.bUpper)
        #f.write("%f " % params.priors.rhoU.params[0, :])
        for item in params.priors.lamWs.params[0, :]:
            f.write("%f " % item)
        f.write("\n")

        # write MCMC step, lower bound, upper bound, and prior params
        f.write("%f " % params.mcmc.lamWOswidth[0, 0])
        f.write("%f " % params.priors.lamWOs.bLower)
        f.write("%f " % params.priors.lamWOs.bUpper)
        #f.write("%f " % params.priors.rhoU.params[0, :])
        for item in params.priors.lamWOs.params[0, :]:  
            f.write("%f " % item)
        f.write("\n")
