/*
 *  GPclasses.h
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



class covMat {
  public:
    int type,d2Dealloc,precsPresent;
	float precs, precz;
	int n,m,p;
	gsl_matrix *x,*z;                    // the dataset used
	gsl_matrix *cov;                   // the squared differences
	gsl_vector *lscale;              // the scaling vector
	double *d2;                 // the scaled distance squared

	~covMat() {
	  gsl_matrix_free(cov);
	  if (d2Dealloc) delete d2;
	}
    covMat(int nIn) { // as a place to hold a cov, by type
	   d2Dealloc=0;
	   type=1;
	   n=nIn;
	   cov=gsl_matrix_calloc(n,n);
	}
    covMat(gsl_matrix *xIn, gsl_vector *lscaleIn, double preczIn, double precsIn) {
	  precsPresent=1;
	  precs=precsIn;
	  initType1(xIn,lscaleIn,preczIn); 
	}
    covMat(gsl_matrix *xIn, gsl_vector *lscaleIn, double preczIn) {
	  precsPresent=0;
	  initType1(xIn,lscaleIn,preczIn);
	 }
	void initType1(gsl_matrix *xIn, gsl_vector *lscaleIn, double preczIn) {
	   d2Dealloc=1;
	   type=1;
	   x=xIn;
	   lscale=lscaleIn;
	   precz=preczIn;
	   n=x->size1; p=x->size2;
	   	   
	   // pre-compute the squared marginal dists
	   d2=new double[p*n*(n-1)/2];
	   computeDist();
	   cov=gsl_matrix_calloc(n,n);
	   computeCov();
	}
    covMat(gsl_matrix *xIn, gsl_matrix *zIn, gsl_vector *lscaleIn, double preczIn) {
	   d2Dealloc=1;
	   type=2;
	   x=xIn; z=zIn;
	   lscale=lscaleIn;
	   precz=preczIn;
	   n=x->size1; p=x->size2; m=z->size1;
	   
	   if (p!=z->size2) { 
	     fprintf(stderr,"Matrix size discrepancy in covMat constructor\n"); 
	     fprintf(stderr,"Matrix 1 size %dx%d\n",int(x->size1),int(x->size2)); 
	     fprintf(stderr,"Matrix 2 size %dx%d\n",int(z->size1),int(z->size2)); 
		 return;
	   }
	   
	   // pre-compute the squared marginal dists
	   d2=new double[p*n*m];
	   computeDist();
	   cov=gsl_matrix_calloc(n,m);
	   computeCov();
	 }
    void computeDist() {
	   int dind=0;
	   double tdub;
	   if (type==1) {
	     for (int ii=0; ii<n-1; ii++)
	       for (int jj=ii+1; jj<n; jj++)
		     for (int kk=0; kk<p; kk++) {
		         tdub=gsl_matrix_get(x,ii,kk) - gsl_matrix_get(x,jj,kk);
			     d2[dind++]=tdub*tdub;
	     }
	   }
	   if (type==2) {
	     for (int ii=0; ii<n; ii++)
	       for (int jj=0; jj<m; jj++)
		     for (int kk=0; kk<p; kk++) {
		       tdub=gsl_matrix_get(x,ii,kk) - gsl_matrix_get(z,jj,kk);
			   d2[dind++]=tdub*tdub;
	     }
	   }
	   //printf("x matrix in covMat:\n");
	   //gsl_matrix_fprintf(stdout,x,"%f");
	}
	void computeCov() {
	   // compute the scaled square dists
	   double td2;
	   int dind=0;
	   if (type==1) {
	     for (int ii=0;ii<(n-1); ii++) {
	       for (int jj=ii+1; jj<n; jj++) {
		     td2=0;
		     for (int kk=0; kk<p; kk++)
	  		   td2+=d2[dind++] * gsl_vector_get(lscale,kk);
		     td2=exp(-1.0*td2) / precz;
	         gsl_matrix_set(cov,ii,jj,td2);
	         gsl_matrix_set(cov,jj,ii,td2);
		   }
	     }
 	     for (int ii=0; ii<n; ii++) {
	       if (precsPresent) gsl_matrix_set(cov,ii,ii,1.0/precz + 1.0/precs);
	                    else gsl_matrix_set(cov,ii,ii,1.0/precz);
	     }
	   }
	   if (type==2) {
	     for (int ii=0;ii<n; ii++) {
	       for (int jj=0; jj<m; jj++) {
		     td2=0;
		     for (int kk=0; kk<p; kk++)
		 	   td2+=d2[dind++] * gsl_vector_get(lscale,kk);
		     td2=exp(-1.0*td2) / precz;
	         gsl_matrix_set(cov,ii,jj,td2);
		   }
 	     }
	   }
	}
};

class GPmodel {
  public:
    gsl_vector *resp; 
    covMat *cov;
	double logLik;
	gsl_vector *tvec;
	gsl_matrix *ch;
	int n;
	int verbose;

  ~GPmodel() {
	gsl_vector_free(tvec);
	gsl_matrix_free(ch);
  }
  GPmodel(gsl_vector *respIn, covMat *covIn) { // data-based constructor
    resp=respIn;
	cov=covIn;
	n=cov->n;
	verbose=0;
	tvec=gsl_vector_alloc(n);
	ch=gsl_matrix_alloc(cov->n,cov->n);
  } 
  void print(FILE *fp) {
    fprintf(fp,"GP Model, size = %d\n",n);
    fprintf(fp,"Response vector: \n");
    for (int ii=0;ii<n; ii++) fprintf(fp,"  %f\n",gsl_vector_get(resp,ii));
    fprintf(fp,"Cov matrix: \n  ");
    for (int ii=0;ii<n; ii++) {
	  for (int jj=0;jj<n;jj++) fprintf(fp,"%f ",gsl_matrix_get(cov->cov,ii,jj));
	  fprintf(fp,"\n");
	}
  }
  double getLogLik() {return logLik;}
  double calcLogLik() {
    verbose=0;
	if (verbose) printf("GPmodel: entered calcLogLok\n");
	gsl_matrix_memcpy(ch,cov->cov);

    if (verbose) {
      printf("\ncov matrix:\n"); 
  	  for (int ii=0; ii<n; ii++) {
	    for (int jj=0; jj<n; jj++) {
	      printf("%8.5f  ",gsl_matrix_get(ch,ii,jj)); 
        }
	    printf("\n");
	  }
    }	

	gsl_linalg_cholesky_decomp(ch);

    if (verbose) {
      printf("\ncholesky matrix:\n"); 
  	  for (int ii=0; ii<n; ii++) {
	    for (int jj=0; jj<n; jj++) {
	      printf("%8.5f  ",gsl_matrix_get(ch,ii,jj)); 
        }
	    printf("\n");
	  }
    }	
	
	double logDet=0;
	for (int ii=0;ii<n; ii++)
	  logDet+=log(gsl_matrix_get(ch,ii,ii));
	
    if (verbose) printf("\nlog det: %8.5f \n",logDet);
	
	gsl_vector_memcpy(tvec,resp);

    // blas routine for solving w/triangular matrix
	gsl_blas_dtrsv(CblasLower,CblasNoTrans,CblasNonUnit,ch,tvec);

    if (verbose) {
      printf("\nresult vector: \n");
  	  for (int ii=0; ii<n; ii++)
	      printf("%8.5f  ",gsl_vector_get(tvec,ii)); 
      printf("\n");
	}

	logLik=-logDet;
	for (int ii=0; ii<cov->n; ii++)
	    logLik-=0.5*gsl_vector_get(tvec,ii)*gsl_vector_get(tvec,ii);

    return logLik;
  }
  void predict(covMat *cPred) {}
};
