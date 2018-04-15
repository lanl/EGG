/*
 * Copyright 2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>

#include <cuda_runtime.h>

#include "cublas_v2.h"
#include "cusolverDn.h"
#include "helper_cuda.h"

//#include "cuChol.h"

#include "helper_cusolver.h"

#include "GPmodel.h"
#include "LAmodel.h"
#include "Prior.h"
#include "LAmodelMCMC.h"

#include <thrust/device_vector.h>
#include <thrust/transform.h>


cusolverDnHandle_t handle = NULL;
cudaStream_t stream = NULL;
cublasHandle_t cuhandle=NULL;
cublasStatus_t cstatus;

extern int timer_sample_betaU;
extern int timer_sample_lamUz;
extern int timer_sample_lamWs;
extern int timer_sample_lamW0;


int chol_cnt = 0;

int GPU;

int main (int argc, char * const argv[]) {


  findCudaDevice(argc, (const char **)argv);

  //cusolverDnHandle_t handle = NULL;
  //cudaStream_t stream = NULL;

  checkCudaErrors(cusolverDnCreate(&handle));
  checkCudaErrors(cudaStreamCreate(&stream));
  checkCudaErrors(cusolverDnSetStream(handle, stream));
  cstatus = cublasCreate(&cuhandle);

    if (cstatus != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
        return EXIT_FAILURE;
    }

   int verbose=0;

  // parse out the command line arguments for infile, outfile, and the number of draws
  int numDraws=0;
  char *inFile, *outFile;
    //const char *inFile="/Users/gatt/Work/projects/saf/Cversion/gpmsa/testModelEta.txt";
    //const char *outFile="/Users/gatt/Work/projects/saf/Cversion/matlab/pvals.txt";
  if (argc>=4) {
    numDraws=atoi(argv[1]);
        inFile=argv[2];
        outFile=argv[3];
  }
  else {
    printf("Invalid command line arguments \n");
        printf("Usage: \n");
        printf("  gpmmcmc <number-of-draws> <model-input-filename> <pvals-output-filename> <boolGPU>\n");
    return -1;
  }
  if(argc==5) GPU=atoi(argv[4]);
  else GPU=1;

  FILE *fp;
  fp=fopen(inFile,"r");
  LAdata *dat=new LAdata(fp);
  if (verbose) printf("Returned from LAdata read constructor\n");
  //getchar();
  LAparams *p0=new LAparams(dat,fp);
  if (verbose) printf("Returned from LAparams read constructor\n");
  //getchar();
  LAmodel *m0=new LAmodel(dat,p0);
  if (verbose) printf("Returned from LAmodel constructor\n");
  //getchar();
  LAmodelMCMC LAM(m0,fp);
  if (verbose) printf("Returned from LAmodelMCMC read constructor\n");
  //getchar();
  fclose(fp);

  printf("Completed read of model with n=%d, m=%d, p=%d, q=%d, pv=%d, pu=%d,\n",
          dat->n, dat->m, dat->p, dat->q, dat->pv, dat->pu);

  // these are not just printing, but also doing initial calculations. 
  printf("Input model log likelihood= %f\n",m0->computeLogLik());
  printf("Input model log prior = %f \n",LAM.computeLogPrior());

  fp=fopen(outFile,"w");
  LAM.writePvalsHeader(fp);
  //LAM.m2=new LAmodel(LAM.mod,100,1);
  int timer=clock();
  printf("Starting sample draws: \n");
  for (int ii=0; ii<numDraws; ii++) {
    LAM.sampleRound();
    LAM.writePvals(fp);
    printf("Round %2d, lik=%10.4f, prior=%10.4f \n", ii, LAM.mod->getLogLik(),LAM.computeLogPrior());
  }
  printf("completed in %f\n",1.0*(clock()-timer)/CLOCKS_PER_SEC);
  printf("total chol calls:  %d\n",chol_cnt);
  printf("betaU sample completed in %f\n",1.0*(timer_sample_betaU)/CLOCKS_PER_SEC);
  printf("lamUz sample completed in %f\n",1.0*(timer_sample_lamUz)/CLOCKS_PER_SEC);
  printf("lamWs sample completed in %f\n",1.0*(timer_sample_lamWs)/CLOCKS_PER_SEC);
  printf("lamW0s sample completed in %f\n",1.0*(timer_sample_lamW0)/CLOCKS_PER_SEC);

  fclose(fp);

  if (handle) { checkCudaErrors(cusolverDnDestroy(handle)); }
  if (stream) { checkCudaErrors(cudaStreamDestroy(stream)); }
  /* Shutdown */
  cstatus = cublasDestroy(cuhandle);

  return 0;
}
