/*
 *  emu.c
 *  
 *
 *  Created by Earl Lawrence on 5/18/2017.
 *  This is a generic emu meant to use an autmatically generated params.h
 *
 *
 *  
 */


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <time.h>

#include "params.h"

// Initialization function that computes the Kriging basis
// Some day, we might just want to store the Kriging basis instead of all of the parameters.
void emuInit() {
  int i,j,k,l;
  double cov;
  gsl_matrix *SigmaSim = gsl_matrix_alloc(m,m);
  gsl_vector *b = gsl_vector_alloc(m);
  
  // Do these one principal component at a time
  for(i=0; i<peta; i++) {
    // Fill in the covariance matrix for the principals components
    // Also make a gsl_vector with the weights.
    for(j=0; j<m; j++) {
      // Diagonal
      gsl_matrix_set(SigmaSim, j, j, (1.0/lamz[i]) + (1.0/lamws[i]));
      // Off-diagonals
      for(k=0; k<j; k++) {
        // Compute the covariance
        cov = 0.0;
        for(l=0; l<p; l++) {
          cov -= beta[i][l]*pow(x[j][l]-x[k][l], 2.0);
        }
        cov = exp(cov)/lamz[i];
        gsl_matrix_set(SigmaSim, j, k, cov);
        gsl_matrix_set(SigmaSim, k, j, cov);
      } // for(k=0; k<j; k++)
      gsl_vector_set(b, j, w[i][j]);
    } // for(j=0; j<m; j++)
    
    // Cholesky and solve
    gsl_linalg_cholesky_decomp(SigmaSim); // takes ~0.177 seconds
    gsl_linalg_cholesky_svx(SigmaSim, b); // takes ~0.001 seconds
    
    // Copy into the Kriging Basis 
    for(j=0; j<m; j++) {
      KrigBasis[i][j] = gsl_vector_get(b, j);
    }
    
    // Save Cholesky decomposition of SigmaSim for PC i
    for(j=0; j<m; j++) {
      for(k=0; k<m; k++) {
        cholSigmaSim[i][j][k] = gsl_matrix_get(SigmaSim, j, k);
      }
    }
    
  } // for(i=0; i<peta; i++)
  gsl_matrix_free(SigmaSim);
  gsl_vector_free(b);
}

// Function that computes the VarBasis (must be run for each new input)
void emuVar(double (*Sigmastar)[m]) {
  int i,j,k;
  gsl_matrix *SigmaSim = gsl_matrix_alloc(m,m);
  gsl_vector *c = gsl_vector_alloc(m); 
  
  // Do these one principal component at a time
  for(i=0; i<peta; i++) {
    // Read in Cholesky decomposition of SigmaSim for PC i
    // Also make a gsl_vector with the weights.
    for(j=0; j<m; j++) {
      gsl_vector_set(c, j, Sigmastar[i][j]); // make a gsl_vector with the weights
      for(k=0; k<m; k++) {
        gsl_matrix_set(SigmaSim, j, k, cholSigmaSim[i][j][k]);
      }
    }
    // Solve using Cholesky decomposition
    gsl_linalg_cholesky_svx(SigmaSim, c); // takes ~0.001 seconds 
    
    // Copy into the Variance Basis
    for(j=0; j<m; j++) {
      VarBasis[i][j] = gsl_vector_get(c, j); 
    }
  } // for(i=0; i<peta; i++)
  gsl_matrix_free(SigmaSim);
  gsl_vector_free(c); 
}

// The actual emulation
void emu(gsl_rng *r, double *xstar, double *ystar, double *ysd, double *wstar, double *wvarstar, double *ysamp) { 
  static int inited=0;
  int i, j, k, s;
  double wsdstar[peta];
  double Sigmastar[peta][m], Rnew[peta], logc;
  double xstarstd[p];
  double wsamp[peta][samp];
  double sumDiag;
  
  // Check the inputs to make sure we're interpolating.
  for(i=0; i<p; i++) {
    if((xstar[i] < xmin[i]) || (xstar[i] > xmax[i])) {
      //printf("The inputs are outside the domain of the emulator.\n");
      printf("Parameter %i must be between %f and %f.\n", i, xmin[i], xmax[i]);
      exit(1);
    }
  } // for(i=0; i<p; i++)
  
  // Standardize the inputs
  for(i=0; i<p; i++) {
    xstarstd[i] = (xstar[i] - xmin[i]) / xrange[i];
  }
  
  // Compute the covariances between the new input and sims for all PCs
  for(i=0; i<peta; i++) {
    for(j=0; j<m; j++) {
      logc = 0.0;
      for(k=0; k<p; k++) {
        logc -= beta[i][k]*pow(x[j][k]-xstarstd[k], 2.0);
      }
      Sigmastar[i][j] = exp(logc)/lamz[i];
    }
  }
  
  // Iinitialize if necesarry
  if(inited==0) {
    emuInit();
    inited=1;
  }
  
  // Run this for every input, since you have to compute VarBasis for each
  emuVar(Sigmastar);
  
  // Compute the covariances between the new inputs for all PCs
  // (since we deal with one input at a time this is a scalar for each PC)
  for(i=0; i<peta; i++) {
    Rnew[i] = (1.0/lamz[i]) + (1.0/lamws[i]);
  }
  
  // Compute wstar, the expected PC weights for the new input
  // and wstarvar, the variance of the PC weights for the new input
  for(i=0; i<peta; i++) {
    wstar[i]=0.0;
    wvarstar[i]=Rnew[i];
    for(j=0; j<m; j++) {
      wstar[i] += Sigmastar[i][j] * KrigBasis[i][j];
      wvarstar[i] -= Sigmastar[i][j] * VarBasis[i][j];
    }
  }
  
  // Sample from the distribution of w for each PC 
  for(i=0; i<peta; i++) {
    wsdstar[i] = pow(wvarstar[i], 0.5);
    for(s=0; s<samp; s++){
      // need to add mean explicitly since we generate a mean 0 Gaussian
      // note that sigma input is standard deviation (not variance)
      wsamp[i][s] = wstar[i] + gsl_ran_gaussian(r, wsdstar[i]);
    }
  }
  
  // Compute ystar, the mean output
  for(i=0; i<neta; i++) {
    ystar[i] = 0.0;
    for(j=0; j<peta; j++) {
      ystar[i] += K[i][j]*wstar[j];
    }
    ystar[i] = ystar[i]*sd + mean[i];
  }
  
  // Only do this if samp!=0
  if(samp != 0) {
    // Compute ysamp, the matrix of sampled y vectors
    for(i=0; i<neta; i++) {
      for(s=0; s<samp; s++){
        ysamp[i*samp+s] = 0.0; 
        for(j=0; j<peta; j++) {
          ysamp[i*samp+s] += K[i][j]*wsamp[j][s];
        }
        ysamp[i*samp+s] = ysamp[i*samp+s]*sd + mean[i];
      }
    }
  } // if(samp != 0)
  
  // Get standard deviation of y at each index (can use to make 95% CI)
  for(i=0; i<neta; i++) {
    sumDiag = 0.0;
    for(j=0; j<peta; j++) {
      sumDiag += wvarstar[j] * pow(K[i][j], 2);
    }
    ysd[i] = sd * pow(sumDiag, 0.5);
  }
  
}

/* 

This main function provides an interface for running the emu at the command line by reading inputs from a file.
It is mostly meant as a demonstration.

*/

int main(int argc, char **argv) {
  
  double xstar[p];
  double ystar[neta];
  double ysd[neta];
  double wstar[peta], wvarstar[peta];
  double pointvar;
  double *ysamp = malloc(neta * samp * sizeof(double));
  int i, j, s;
  FILE *infile;
  FILE *outfile;
  FILE *outfileysd;
  FILE *outfilewstar;
  FILE *outfilewvarstar;
  FILE *outfilesamp;
  FILE *outfilepointvar;
  int colchars = 30;
  char instring[p * colchars];
  char outname[p * colchars];
  char *token;
  int good = 1;
  int ctr = 0;
  char ctrc[100]; 
  // Initialize and set up "random" variable
  gsl_rng *r = gsl_rng_alloc(gsl_rng_taus2); // GSL's Taus generator
  gsl_rng_set(r, time(NULL)); // Initialize the GSL generator with time
  
  // Read inputs from a file
  // File should be space delimited with p numbers on each line.
  if((infile = fopen("xstar.dat","r"))==NULL) {
    printf("Cannot find inputs.\n");
    exit(1);
  }
  
  // Save the outputs in file.
  // Each row is the output for the corresponding row in xstar.dat
  if ((outfile = fopen("ystar.dat","w"))==NULL) {
    printf("Cannot open ystar.dat.\n");
    exit(1);
  }
  
  // Save the std deviation of y outputs in file.
  // Each row is the output for the corresponding row in xstar.dat
  if ((outfileysd = fopen("ysd.dat","w"))==NULL) {
    printf("Cannot open ysd.dat.\n");
    exit(1);
  }
  
  if(samp != 0) {
    // Save the sample outputs in file.
    // Each row is the output for one sample of the corresponding row in xstar.dat (in order)
    if ((outfilesamp = fopen("ysamp.dat","w"))==NULL) {
      printf("Cannot open ysamp.dat.\n");
      exit(1);
    }
  }
  
  // Save the expected weight outputs in file.
  // Each row is the output the corresponding row in xstar.dat
  if ((outfilewstar = fopen("wstar.dat","w"))==NULL) {
    printf("Cannot open wstar.dat.\n");
    exit(1);
  }
  
  // Save the variance of weight outputs in file.
  // Each row is the output the corresponding row in xstar.dat
  if ((outfilewvarstar = fopen("wvarstar.dat","w"))==NULL) {
    printf("Cannot open wvarstar.dat.\n");
    exit(1);
  }
  
  // Save the pointvar (covariance matrix sum of diagonals) outputs in file.
  // Each row is the output the corresponding row in xstar.dat
  if ((outfilepointvar = fopen("ypointvar.dat","w"))==NULL) {
    printf("Cannot open ypointvar.dat.\n");
    exit(1);
  }
  
  // Read in the inputs and emulate the results.
  while(good == 1) {
    
    // Read each line
    if(fgets(instring, p * colchars, infile) != NULL) {
      token = strtok(instring, " ");
      
      // Parse each line, which is space delimited
      for(i=0; i<p; i++) {
        xstar[i] = atof(token);
        token = strtok(NULL, " ");
      }
      
      // Get the answers (expectation and samples).
      emu(r, xstar, ystar, ysd, wstar, wvarstar, ysamp); 
      
      // Write the answer (expectation) and standard deviation at each index point
      for(i=0; i<neta; i++) {
        fprintf(outfile, "%f ", ystar[i]);
        fprintf(outfileysd, "%f ", ysd[i]);
      }
      fprintf(outfile, "\n");
      fprintf(outfileysd, "\n");
      
      if(samp != 0) {
        // Write the answer (samples)
        // each row is its own sample, sampled samp times for each row of x
        for(s=0; s<samp; s++) {
          for(i=0; i<neta; i++) {
            fprintf(outfilesamp, "%f ", ysamp[i*samp+s]);
          }
          fprintf(outfilesamp, "\n");
        }
        fprintf(outfilesamp, "\n");
      }
      
      // Write the answer (basis expectation wstar and variance wvarstar)
      for(i=0; i<peta; i++) {
        fprintf(outfilewstar, "%f ", wstar[i]);
        fprintf(outfilewvarstar, "%f ", wvarstar[i]);
      }
      fprintf(outfilewstar, "\n");
      fprintf(outfilewvarstar, "\n");
      
      // Calculate variance point estimate
      pointvar = 0.0;
      for(i=0; i<neta; i++) {
        pointvar += pow(ysd[i], 2);
      }
      // Write the answer (variance estimate)
      fprintf(outfilepointvar, "%f ", pointvar);
      fprintf(outfilepointvar, "\n");
      
      ctr++;
    } else {
      good = 0;
    }
  }
  fclose(infile);
  fclose(outfile);
  fclose(outfileysd);
  if(samp != 0) { fclose(outfilesamp); }
  fclose(outfilewstar);
  fclose(outfilewvarstar);
  fclose(outfilepointvar);
  
  // Free memory
  gsl_rng_free(r);
  free(ysamp);
  
}
