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

static int timer_sample_betaU=0;
static int timer_sample_lamUz=0;
static int timer_sample_lamWs=0;
static int timer_sample_lamW0=0;

extern int chol_cnt;

class LAmodelMCMC {
  public:
    // inputs
	  LAmodel *mod, *m2;
	  struct MCMCparams {
	    Prior *prior;
	    double width;
	  };
	  int n,m,p,q,pu,pv;
      MCMCparams theta, betaU, betaV, lamVz, lamUz, lamWs, lamWOs, lamOs;
	  int verbose;

 LAmodelMCMC (LAmodel *min, FILE *fp) { 
    verbose=0;
    mod=min;
	n=mod->n; m=mod->m; p=mod->p; q=mod->q; pu=mod->pu; pv=mod->pv;
	if (n>0) readPriorMCMC(fp,&theta,1);
	if (n>0) readPriorMCMC(fp,&betaV,4);
	readPriorMCMC(fp,&betaU,4);
	if (n>0) readPriorMCMC(fp,&lamVz,2);
	readPriorMCMC(fp,&lamUz,2);
	readPriorMCMC(fp,&lamWs,2);
	readPriorMCMC(fp,&lamWOs,2);
	if (n>0) readPriorMCMC(fp,&lamOs,2);

    if (n>0) {
	  if (verbose) printf("theta  %8.4f %8.4f %8.4f %8.4f %8.4f \n",theta.width,theta.prior->lBound,theta.prior->uBound,
             theta.prior->params[0],theta.prior->params[1]);
	  if (verbose) printf("betaV   %8.4f %8.4f %8.4f %8.4f %8.4f \n",betaV.width,betaV.prior->lBound,betaV.prior->uBound,
             betaV.prior->params[0],betaV.prior->params[1]);
	  if (verbose) printf("lamVz  %8.4f %8.4f %8.4f %8.4f %8.4f \n",lamVz.width,lamVz.prior->lBound,lamVz.prior->uBound,
             lamVz.prior->params[0],lamVz.prior->params[1]);
	  if (verbose) printf("lamOs  %8.4f %8.4f %8.4f %8.4f %8.4f \n",lamOs.width,lamOs.prior->lBound,lamOs.prior->uBound,
             lamOs.prior->params[0],lamOs.prior->params[1]);
    }
	if (verbose) printf("betaU   %8.4f %8.4f %8.4f %8.4f %8.4f \n",betaU.width,betaU.prior->lBound,betaU.prior->uBound,
             betaU.prior->params[0],betaU.prior->params[1]);
	if (verbose) printf("lamUz  %8.4f %8.4f %8.4f %8.4f %8.4f \n",lamUz.width,lamUz.prior->lBound,lamUz.prior->uBound,
             lamUz.prior->params[0],lamUz.prior->params[1]);
	if (verbose) printf("lamWs  %8.4f %8.4f %8.4f %8.4f %8.4f \n",lamWs.width,lamWs.prior->lBound,lamWs.prior->uBound,
             lamWs.prior->params[0],lamWs.prior->params[1]);
	if (verbose) printf("lamWOs %8.4f %8.4f %8.4f %8.4f %8.4f \n",lamWOs.width,lamWOs.prior->lBound,lamWOs.prior->uBound,
             lamWOs.prior->params[0],lamWOs.prior->params[1]);
	
 }
 ~LAmodelMCMC() {
   if (n>0) delete theta.prior; 
   if (n>0) delete betaV.prior; 
   delete betaU.prior; 
   if (n>0) delete lamVz.prior; 
   delete lamUz.prior; 
   delete lamWs.prior; 
   delete lamWOs.prior; 
   if (n>0) delete lamOs.prior; 
 }
 void readPriorMCMC(FILE *fp, MCMCparams *mp, int ptype) {
   float fin,fin1,fin2,fin3,fin4;
   fscanf(fp,"%f",&fin); mp->width=fin;
   fscanf(fp,"%f",&fin3); 
   fscanf(fp,"%f",&fin4); 
   fscanf(fp,"%f",&fin1); 
   fscanf(fp,"%f",&fin2); 
   mp->prior=new Prior(fin1,fin2,ptype,fin3,fin4);   
 }
 
 void writePvalsHeader(FILE *fp) {
   fprintf(fp,"%d %d %d %d %d %d\n",n,m,p,q,pv,pu);
 }
 void writePvals(FILE *fp) {
	  if (n>0) {
	    for (int ii=0;ii<q;ii++) { fprintf(fp,"%f ",gsl_vector_get(mod->pp->theta,ii)); }
	      fprintf(fp,"\n");
	    for (int ii=0; ii<1; ii++) for (int jj=0; jj<p; jj++) 
	      { fprintf(fp,"%f ",gsl_vector_get(mod->pp->betaV[ii],jj)); }
	      fprintf(fp,"\n");
	    for (int ii=0; ii<1; ii++) { fprintf(fp,"%f ",gsl_vector_get(mod->pp->lamVz,ii)); }
	      fprintf(fp,"\n");
	    fprintf(fp,"%f\n",mod->pp->lamOs);
	  }
	  for (int ii=0; ii<pu; ii++) for (int jj=0; jj<(p+q); jj++) 
	    { fprintf(fp,"%f ",gsl_vector_get(mod->pp->betaU[ii],jj)); }
	    fprintf(fp,"\n");
	  for (int ii=0; ii<pu; ii++) { fprintf(fp,"%f ",gsl_vector_get(mod->pp->lamUz,ii)); }
	    fprintf(fp,"\n");
	  for (int ii=0; ii<pu; ii++) { fprintf(fp,"%f ",gsl_vector_get(mod->pp->lamWs,ii)); }
	    fprintf(fp,"\n");
	  fprintf(fp,"%f\n",mod->pp->lamWOs);
	  fprintf(fp,"%f\n",mod->getLogLik());
	  fprintf(fp,"%f\n",computeLogPrior(0));
	  fprintf(fp,"\n");
 }
 
 void sampleRound() {
   	double newVal, aCorr;

	double oldlik, oldprior, old_val, sig_loglik, newlik, newprior, old_precz, old_precs; 
	double *old_precs_arr = new double[pu];
	double *old_sig_loglik_arr = new double[pu];
   
   // Sample betaU
   int timer1=clock();
   for (int ii=0; ii<pu; ii++) 
     for (int jj=0; jj<(p+q); jj++) 
    {
	   old_val = gsl_vector_get(mod->pp->betaU[ii],jj);
	   newVal=exp(-0.25* old_val) + (urand() - 0.5) * betaU.width;
	   newVal=-4*log(newVal);
	   if (verbose) printf("betaU %2d-%2d: Updating with value %f\n",ii,jj,newVal);
	   if (betaU.prior->withinBounds(newVal)) 
	{
	     	oldlik = mod->getLogLik();
		oldprior = betaU.prior->computeLogPrior(old_val);
		sig_loglik = mod->SigWGP[ii]->logLik;
	     	mod->updateBetaU(ii,jj,newVal);
	     	newlik = mod->computeLogLik();
		newprior = betaU.prior->computeLogPrior(gsl_vector_get(mod->pp->betaU[ii],jj));
	     if (!accept(newlik, newprior, oldlik, oldprior, 1) ) 
		{
	     		gsl_vector_set(mod->pp->betaU[ii],jj,old_val);
			mod->SigWGP[ii]->logLik=sig_loglik;
	     		mod->computeLogLik();
	   	}
	 }
   }


   int timer2=clock();
   timer_sample_betaU+=timer2-timer1;
   if (verbose) printf("Completed sampling betaU, logPrior=%f\n",computeLogPrior(0));

   // Sample lamUz
   timer1=clock();
   for (int ii=0; ii<pu; ii++) {
	 aCorr=chooseVal(gsl_vector_get(mod->pp->lamUz,ii),&newVal);
	 if (verbose) printf("lamUz %2d: Updating with value %f\n",ii,newVal);
	 if (aCorr!=0) {
	   if (lamUz.prior->withinBounds(newVal)) {
	     	oldlik = mod->getLogLik();
	     	old_val = gsl_vector_get(mod->pp->lamUz,ii);
		oldprior = lamUz.prior->computeLogPrior(old_val);
		sig_loglik = mod->SigWGP[ii]->logLik;
		old_precz = mod->SigWc[ii]->precz;
	     	mod->updateLamUz(ii,newVal);
		newlik = mod->computeLogLik();
		newprior = lamUz.prior->computeLogPrior(gsl_vector_get(mod->pp->lamUz,ii));
	     if (!accept(newlik, newprior, oldlik, oldprior, aCorr) ) 
		{
	     		gsl_vector_set(mod->pp->lamUz,ii,old_val);
			mod->SigWc[ii]->precz = old_precz;
			mod->SigWGP[ii]->logLik=sig_loglik;
	     		mod->computeLogLik();
		}
	   }
     }
   }
   timer2=clock();
   timer_sample_lamUz+=timer2-timer1;
   if (verbose) printf("Completed sampling lamUz, logPrior=%f\n",computeLogPrior(0));

   // Sample lamWs
   timer1=clock();
   for (int ii=0; ii<pu; ii++) {
	 aCorr=chooseVal(gsl_vector_get(mod->pp->lamWs,ii),&newVal);
	 if (verbose) printf("lamWs %2d: Updating with value %f\n",ii,newVal);
	 if (aCorr!=0) {
	   if (lamWs.prior->withinBounds(newVal)) {
	     	oldlik = mod->getLogLik();
	      	old_val = gsl_vector_get(mod->pp->lamWs,ii);
		oldprior = lamWs.prior->computeLogPrior(old_val);
		sig_loglik = mod->SigWGP[ii]->logLik;
		old_precs = mod->SigWc[ii]->precs;
		mod->updateLamWs(ii,newVal);
		newlik = mod->computeLogLik();
		newprior = lamWs.prior->computeLogPrior(gsl_vector_get(mod->pp->lamWs,ii));
	     if (!accept(newlik, newprior, oldlik, oldprior, aCorr) ) 
		{
	     		gsl_vector_set(mod->pp->lamWs,ii,old_val);
			mod->SigWc[ii]->precs = old_precs;
			mod->SigWGP[ii]->logLik=sig_loglik;
	     		mod->computeLogLik();
	     	}
	   }
     }
   }
   timer2=clock();
   timer_sample_lamWs+=timer2-timer1;
   if (verbose) printf("Completed sampling lamWs, logPrior=%f\n",computeLogPrior(0));
   // Sample lamWOs
   timer1=clock();
	 aCorr=chooseVal(mod->pp->lamWOs,&newVal);
	 if (verbose) printf("lamWOs: Updating with value %f\n",newVal);
	 if (aCorr!=0) {
	   if (lamWOs.prior->withinBounds(newVal)) {
	     	oldlik = mod->getLogLik();
	     	old_val = mod->pp->lamWOs;
		oldprior = lamWOs.prior->computeLogPrior(old_val);
		for (int ind=0; ind<pu; ind++) 
		{
			old_precs_arr[ind] = mod->SigWc[ind]->precs;
			old_sig_loglik_arr[ind] = mod->SigWGP[ind]->logLik;
		}
		mod->updateLamWOs(newVal);
		newlik = mod->computeLogLik();
		newprior =  lamWOs.prior->computeLogPrior(mod->pp->lamWOs);
	     if (!accept(newlik, newprior, oldlik, oldprior, aCorr) ) 
		{
			//m2->updateLamWOs(old_val);
	     		mod->pp->lamWOs = old_val;
			for (int ind=0; ind<pu; ind++)
                	{	
				mod->SigWc[ind]->precs = old_precs_arr[ind];
				mod->SigWGP[ind]->logLik= old_sig_loglik_arr[ind];
			}
	     		mod->computeLogLik();
	     	}
	   }
     }
   timer2=clock();
   timer_sample_lamW0+=timer2-timer1;
   if (verbose) printf("Completed sampling lamWOs, logPrior=%f\n",computeLogPrior(0));   // Sample lamOs

}

 double urand() {
   return rand()*1.0/RAND_MAX;
 }

 double chooseVal(double cval, double *dval) { // choose a new value in an interval around the current value
   // select the interval, and draw a new value.
   double w=cval/3; if (w<1) w=1;
   *dval=cval + (urand()*2-1)*w;
   // do a correction, which depends on the old and new interval
   double w1=*dval/3; if (w1<1) w1=1;
   double acorr;
   if (cval > (*dval+w1))
     acorr=0;
   else
     acorr=w/w1;
   //printf("cval=%10.4f, dval=%10.4f, acorr=%10.4f\n",cval,*dval,acorr);
   return acorr;
 }

 int accept(double newLik, double newPrior, double oldLik, double oldPrior, double correction) {
   double testVal;
   testVal=log(urand());
   //verbose = 1;
   if (verbose)
      printf("MCMC (%10.4f<%10.4f)? to newLik=%10.4f,newPrior=%10.4f to oldLik=%10.4f,oldPrior=%10.4f, corr=%10.4f ... ",
	         testVal, newLik+newPrior-oldLik-oldPrior+log(correction), newLik, newPrior, oldLik, oldPrior, correction);
   verbose = 0;
   //getchar();
   if (testVal < ( (newLik + newPrior) - (oldLik + oldPrior) + log(correction) ) ) {
      if (verbose) printf("accept.\n");
      return 1;
	}
	else {
      if (verbose) printf("reject.\n");
	  return 0;
	}
 }
 
 double computeLogPrior() {return computeLogPrior(0);}
 double computeLogPrior(int verbose) { 
   double lp=0;
   if (n>0) { 
	 lp+=theta.prior->computeLogPrior(mod->pp->theta);    if (verbose) printf("theta %f\n",lp);
	 lp+=betaV.prior->computeLogPrior(mod->pp->betaV,1);  if (verbose) printf("betaV  %f\n",lp);
     lp+=lamOs.prior->computeLogPrior(mod->pp->lamOs);    if (verbose) printf("lamOs %f\n",lp);
	 lp+=lamVz.prior->computeLogPrior(mod->pp->lamVz);    if (verbose) printf("lamVz %f\n",lp);
    }
   lp+=betaU.prior->computeLogPrior(mod->pp->betaU,pu); if (verbose) printf("betaU %f\n",lp);
   lp+=lamUz.prior->computeLogPrior(mod->pp->lamUz);    if (verbose) printf("lamUz %f\n",lp);
   lp+=lamWs.prior->computeLogPrior(mod->pp->lamWs);    if (verbose) printf("lamWs %f\n",lp);
   lp+=lamWOs.prior->computeLogPrior(mod->pp->lamWOs);  if (verbose) printf("lamWOs %f\n",lp);
   
   return lp;
 }
};


