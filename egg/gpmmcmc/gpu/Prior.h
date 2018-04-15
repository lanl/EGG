/*
 *  LAmodelMCMC.h
 *  gpmsa
 *
 *  Created by gatt on 4/2/08.
 *  Copyright 2008 LANL. All rights reserved.
 *
 */
#include <iostream>
#include <stdio.h>
#include <math.h>

#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

class Prior {
 public:
   double params[2];
   int ptype;
   double lBound, uBound;
 
 Prior(double param1in, double param2in, int ptypeIn, double lbin, double ubin) {
   params[0]=param1in; params[1]=param2in;
   ptype=ptypeIn;
   lBound=lbin;
   uBound=ubin;
 }
 
 double computeLogPrior(gsl_vector *vec) {
   double ll=0;
   for (int ii=0; ii<vec->size; ii++)
      ll+=computeLogPrior(gsl_vector_get(vec,ii));
   return ll;
 }
	   
 double computeLogPrior(gsl_vector **vec,int size) {
   double ll=0;
   for (int ii=0; ii<size; ii++)
     for (int jj=0; jj<vec[ii]->size; jj++)
      ll+=computeLogPrior(gsl_vector_get(vec[ii],jj));
   return ll;
 }

 int withinBounds(double val) {
   if (val<lBound | val>uBound) return 0;
   else return 1;
 }
   
 double computeLogPrior(double val) {
   double lp;
   switch (ptype) {
   case 1: { 
	   // Normal
	   double tres=(val-params[0])/params[1];
       lp=-.5 * tres*tres ;
	    //printf("Normal prior %10.4f, val=%10.4f, params1=%10.4f, params2=%10.4f\n",lp,val,params[0],params[1]);
     }
     break;
     case 2:
	   // Gamma
	   lp= (params[0]-1)*log(val) - params[1]*val;
     break;
     case 3:
	   // Beta
 	   lp= (params[0]-1)*log(val) + (params[1]-1)*log(1-val);
	   break;
     case 4:
	   // Beta, under domain transformation
	   val=exp(-0.25*val);
 	   lp= (params[0]-1)*log(val) + (params[1]-1)*log(1-val);
	   break;
	  default:
	    fprintf(stderr,"Big Touble in Prior.h\n");
   } 
   return lp;
 } 
};

