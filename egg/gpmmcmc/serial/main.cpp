#include <iostream>
#include <ctime>
#include <stdlib.h>
#include "GPmodel.h"
#include "LAmodel.h"
#include "Prior.h"
#include "LAmodelMCMC.h"

int main (int argc, char * const argv[]) {
   int verbose=0;
   
  // parse out the command line arguments for infile, outfile, and the number of draws
  int numDraws=0;
  char *inFile, *outFile;
    //const char *inFile="/Users/gatt/Work/projects/saf/Cversion/gpmsa/testModelEta.txt";
    //const char *outFile="/Users/gatt/Work/projects/saf/Cversion/matlab/pvals.txt";
  if (argc==4) {
    numDraws=atoi(argv[1]);
	inFile=argv[2];
	outFile=argv[3];
  }
  else {
    printf("Invalid command line arguments \n");
	printf("Usage: \n");
	printf("  gpmmcmc <number-of-draws> <model-input-filename> <pvals-output-filename> \n");
    return -1;
  }
  
  FILE *fp;
  fp=fopen(inFile,"r");
  LAdata *dat=new LAdata(fp);
  if (verbose) printf("Returned from LAdata read constructor\n");
  LAparams *p0=new LAparams(dat,fp);
  if (verbose) printf("Returned from LAparams read constructor\n");
  LAmodel *m0=new LAmodel(dat,p0);
  if (verbose) printf("Returned from LAmodel constructor\n");
  LAmodelMCMC LAM(m0,fp);
  if (verbose) printf("Returned from LAmodelMCMC read constructor\n");
  fclose(fp);

  printf("Completed read of model with n=%d, m=%d, p=%d, q=%d, pv=%d, pu=%d,\n",
          dat->n, dat->m, dat->p, dat->q, dat->pv, dat->pu);

  // these are not just printing, but also doing initial calculations. 
  printf("Input model log likelihood= %f\n",m0->computeLogLik());
  printf("Input model log prior = %f \n",LAM.computeLogPrior());

  fp=fopen(outFile,"w");
  LAM.writePvalsHeader(fp);
  int timer=clock();
  for (int ii=0; ii<numDraws; ii++) {
    LAM.sampleRound();
	LAM.writePvals(fp);
	printf("Round %2d, lik=%10.4f, prior=%10.4f \n", ii,
	       LAM.mod->getLogLik(),LAM.computeLogPrior());
  }
  printf("completed in %f\n",1.0*(clock()-timer)/CLOCKS_PER_SEC);
  fclose(fp);
	
  return 0;
}

