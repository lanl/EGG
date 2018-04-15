/*
 *  LAmodel.h
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

class LAdata {
public:
      gsl_matrix *x, *zt;
	  gsl_vector *vu, **w;
  	  int n,m,p,q,pv,pu;
  LAdata(FILE *fp) {
  // Read in the critical variables from the model initialization, set up precomputed structs
    int verbose=0;
    float fin;
	//   scalar: n, m, p, q, pv, pu
	//   matrix: x, zt
	//   vector: vu
	//   vector array: w

	// model size parms: n, m, p, q, pv, pu
      fscanf(fp,"%d %d %d %d %d %d",&n, &m, &p, &q, &pv, &pu);
	  if (verbose) printf("LAdata: Completed read of size parameters: \n");
	  if (verbose) printf("LAdata:   n=%d m=%d p=%d q=%d pv=%d pu=%d \n",n,m,p,q,pv,pu);

    // allocate data structures
	  if (n>0) {
	    x=gsl_matrix_alloc(n,p);
	    vu=gsl_vector_alloc(n*(pv+pu));
	  }
	  zt=gsl_matrix_alloc(m,p+q);
	  w=new gsl_vector*[pu];  for (int ii=0; ii<pu; ii++) w[ii]=gsl_vector_alloc(m);
	  if (verbose) printf("LAdata: Completed allocate of datasets: x, zt, vu, w\n");

	// read datasets x, zt, vu, w
	  for (int jj=0; jj<p; jj++) for (int ii=0; ii<n; ii++) 
	    { fscanf(fp,"%f",&fin); gsl_matrix_set(x,ii,jj,fin); }
	  if (verbose & (n>0)) printf("LAdata: Completed read of x, first float %f \n",gsl_matrix_get(x,0,0));
	  for (int jj=0; jj<(p+q); jj++) for (int ii=0; ii<m; ii++) 
	    { fscanf(fp,"%f",&fin); gsl_matrix_set(zt,ii,jj,fin); }
	  if (verbose) printf("LAdata: Completed read of zt, first float %f \n",gsl_matrix_get(zt,0,0));
	  for (int ii=0; ii<n*(pv+pu); ii++) { fscanf(fp,"%f",&fin); gsl_vector_set(vu,ii,fin); }
	  if (verbose & (n>0)) printf("LAdata: Completed read of vu, first float %f \n",gsl_vector_get(vu,0));
	  for (int ii=0; ii<pu; ii++) for (int jj=0; jj<m; jj++) 
	    { fscanf(fp,"%f",&fin); gsl_vector_set(w[ii],jj,fin); }
	  if (verbose) printf("LAdata: Completed read of w, first float %f \n",gsl_vector_get(w[0],0));
  }
  ~LAdata() {
    int verbose=0;
    if (verbose) printf("LAdata Destructor: deleting data structures\n");
	gsl_matrix_free(x);
	gsl_matrix_free(zt);
	gsl_vector_free(vu);
	for (int ii=0; ii<pu; ii++) gsl_vector_free(w[ii]); delete w;
  }

};

class LAparams {
public:
  gsl_vector *theta, *lamVz, *lamUz, *lamWs, *LamSim;
  gsl_vector **betaV, **betaU;
  double lamOs, lamWOs;
  gsl_matrix *SigObs;
  int n,m,p,q,pv,pu;

  LAparams(LAdata *d, FILE *fp)  {
    int verbose=0;
    n=d->n; m=d->m; p=d->p; q=d->q; pv=d->pv; pu=d->pu;
	if (verbose) printf("LAparams: read constructor \n");
	if (verbose) printf("LAparams: n=%d m=%d p=%d q=%d pv=%d pu=%d \n",n,m,p,q,pv,pu);

    float fin;
    //   scalar: lamOs, lamWOs
	//   vector: theta, betaV, lamVz, (array) betaU, lamUz, lamWs, LamSim
	//   matrix: SigObs
	  
    allocate();
	  
	//Read lamOs, lamWOs, theta, betaV, lamVz, (array) betaU, lamUz, lamWs, LamSim, SigObs
	fscanf(fp,"%f",&fin); lamOs=fin;
	fscanf(fp,"%f",&fin); lamWOs=fin;
	if (n>0) {
	  for (int ii=0;ii<q;ii++) { fscanf(fp,"%f",&fin); gsl_vector_set(theta,ii,fin); }
	  if (verbose) printf("LAparams: Completed read of theta\n");
	  for (int ii=0; ii<1; ii++) for (int jj=0; jj<p; jj++) 
	    { fscanf(fp,"%f",&fin);  gsl_vector_set(betaV[ii],jj,fin); }
	  if (verbose) printf("LAparams: Completed read of betaV\n");
	  for (int ii=0; ii<1; ii++) { fscanf(fp,"%f",&fin); gsl_vector_set(lamVz,ii,fin); }
	  if (verbose) printf("LAparams: Completed read of lamVz\n");
	}
	for (int ii=0; ii<pu; ii++) for (int jj=0; jj<(p+q); jj++) 
	  { fscanf(fp,"%f",&fin); gsl_vector_set(betaU[ii],jj,fin); }
	if (verbose) printf("LAparams: Completed read of betaU\n");
	for (int ii=0; ii<pu; ii++) { fscanf(fp,"%f",&fin); gsl_vector_set(lamUz,ii,fin); }
	if (verbose) printf("LAparams: Completed read of lamUz\n");
	for (int ii=0; ii<pu; ii++) { fscanf(fp,"%f",&fin); gsl_vector_set(lamWs,ii,fin); }
	if (verbose) printf("LAparams: Completed read of lamWs\n");
	for (int ii=0; ii<pu; ii++) { fscanf(fp,"%f",&fin); gsl_vector_set(LamSim,ii,fin); }
	if (verbose) printf("LAparams: Completed read of LamSim\n");
	if (n>0) {
  	  for (int jj=0; jj<n*(pv+pu); jj++) for (int ii=0; ii<n*(pv+pu); ii++) 
	    { fscanf(fp,"%f",&fin); gsl_matrix_set(SigObs,ii,jj,fin); }
  	  if (verbose) printf("LAparams: Completed read of SigObs, first float %f \n",gsl_matrix_get(SigObs,0,0));
	}

  }
  LAparams(LAparams *pin) {
    int verbose=0;
    n=pin->n; m=pin->m; p=pin->p; q=pin->q; pv=pin->pv; pu=pin->pu;
	if (verbose) printf("LAparams: copy constructor \n");
	if (verbose) printf("LAparams: n=%d m=%d p=%d q=%d pv=%d pu=%d \n",n,m,p,q,pv,pu);

    allocate();

    // copy model structures
    lamOs=pin->lamOs;
	lamWOs=pin->lamWOs;
	if (n>0) {
      gsl_vector_memcpy(theta,pin->theta);
   	  for (int ii=0; ii<1; ii++)
   	    gsl_vector_memcpy(betaV[ii],pin->betaV[ii]);
	  gsl_vector_memcpy(lamVz,pin->lamVz);
  	  gsl_matrix_memcpy(SigObs,pin->SigObs);
	}
	for (int ii=0; ii<(pu); ii++)
   	  gsl_vector_memcpy(betaU[ii],pin->betaU[ii]);
	gsl_vector_memcpy(lamUz,pin->lamUz);
	gsl_vector_memcpy(lamWs,pin->lamWs);
	gsl_vector_memcpy(LamSim,pin->LamSim);
    if (verbose) printf("LAparams: copy constructor completed\n");
  }
  void allocate() {
	if (n>0) {
      theta=gsl_vector_alloc(q);
	  betaV=new gsl_vector*[1];  for (int ii=0; ii<1; ii++)  betaV[ii]=gsl_vector_alloc(p); 
	  lamVz=gsl_vector_alloc(1); //lamVzGnum
  	  SigObs=gsl_matrix_alloc(n*(pv+pu),n*(pv+pu));
	}
	betaU=new gsl_vector*[pu]; for (int ii=0; ii<pu; ii++) betaU[ii]=gsl_vector_alloc(p+q);
	lamUz=gsl_vector_alloc(pu);
	lamWs=gsl_vector_alloc(pu);
	LamSim=gsl_vector_alloc(pu);
  }
  ~LAparams() {
    int verbose=0;
    if (verbose) printf("LAparams: destructor\n");
	if (n>0) {
      gsl_vector_free(theta);
      for (int ii=0; ii<1; ii++) gsl_vector_free(betaV[ii]);  delete betaV;
      gsl_vector_free(lamVz);
      gsl_matrix_free(SigObs);
	}
    for (int ii=0; ii<pu; ii++) gsl_vector_free(betaU[ii]); delete betaU;
    gsl_vector_free(lamUz);
    gsl_vector_free(lamWs);
    gsl_vector_free(LamSim);
  }
};

class LAmodel {
  public:
    // inputs
	  LAdata *d;
	  LAparams *pp;
  	  int n,m,p,q,pv,pu;
	// internals
      gsl_matrix *xtheta;
  	  covMat *SigVc, **SigUc, **SigWc, **SigUWc, *SigVUgWc;
	  GPmodel **SigWGP, *SigVUgW_GP; 
	  gsl_matrix **W, **SigUgW;
  	  gsl_vector *MuVUgW;
	  double logLik;

 void matrixPrint(FILE *fp, gsl_matrix *mat) {
   for (int ii=0; ii<mat->size1; ii++) {
     for (int jj=0; jj<mat->size2; jj++) fprintf(fp,"%10.4f ",gsl_matrix_get(mat,ii,jj));
     fprintf(fp,"\n");
   }
 }

 LAmodel(LAmodel *min) { 
   int verbose=0;
   // this constructor forms a duplicate model -- copy constructor
   //    datasets are pointed to, model structures are duplicated.
   n=min->n; m=min->m; p=min->p; q=min->q; pv=min->pv; pu=min->pu;
   if (verbose) printf("  n=%d m=%d p=%d q=%d pv=%d pu=%d \n",n,m,p,q,pv,pu);

   logLik=min->logLik;
   d=min->d;
   pp=new LAparams(min->pp);

   // Allocate and precompute data products
    initPrecompute();
 }

 LAmodel(LAdata *inDat, LAparams *inParams) {
     d=inDat;
     n=d->n; m=d->m; p=d->p; q=d->q; pv=d->pv; pu=d->pu;
	 pp=new LAparams(inParams);
     initPrecompute();
  }
  // Build up the precomputed data products.

  void initPrecompute() {
    int verbose=0;
	// allocate and construct the [x theta] matrix
	if (n>0) 
	  xtheta=gsl_matrix_alloc(n,p+q);
	  for (int ii=0; ii<n; ii++) {
	    for (int jj=0; jj<p; jj++)
	      gsl_matrix_set(xtheta,ii,jj,gsl_matrix_get(d->x,ii,jj));
	    for (int jj=0; jj<q; jj++)
	      gsl_matrix_set(xtheta,ii,p+jj,gsl_vector_get(pp->theta,jj));
	}
			
    // Build the SigW cov block diagonal elements
	SigWc=new covMat*[pu];
	for (int ii=0; ii<pu; ii++) {
  	  SigWc[ii]=new covMat(d->zt,pp->betaU[ii],gsl_vector_get(pp->lamUz,ii),
	                      1.0/(1.0/(gsl_vector_get(pp->LamSim,ii)*pp->lamWOs) + 1.0/gsl_vector_get(pp->lamWs,ii)) );
	}
	if (verbose) printf("Completed precomputation of SigWc, it's size is %d\n",int(SigWc[0]->cov->size1));
	
	// logLik of W is the logLik of the independent response GPs.
  	  SigWGP=new GPmodel*[pu];
  	  for (int ii=0; ii<pu; ii++) {
	    SigWGP[ii]=new GPmodel(d->w[ii],SigWc[ii]); 
		  //SigWGP[ii]->print(stdout);
		SigWGP[ii]->calcLogLik();
		  //printf("SigW likelihood %d = %f\n",ii,SigWGP[ii]->getLogLik());
	  }
      if (verbose) printf("Completed precomputation of SigWGP, first float %f\n",
	                       gsl_matrix_get(SigWGP[0]->cov->cov,0,0));
	  
	if (n>0) {
      // build the SigV cov block diagonal element
	    SigVc=new covMat(d->x,pp->betaV[0],gsl_vector_get(pp->lamVz,0));
	    if (verbose) printf("Completed precomputation of SigVc, it's size is %d\n",int(SigVc->cov->size1));
	
      // Build the SigU cov block diagonal elements
	    SigUc=new covMat*[pu];
	    for (int ii=0; ii<pu; ii++)
  	      SigUc[ii]=new covMat(xtheta,pp->betaU[ii],gsl_vector_get(pp->lamUz,ii),gsl_vector_get(pp->lamWs,ii));
	    if (verbose) printf("Completed precomputation of SigUc, it's size is %d\n",int(SigUc[0]->cov->size1));
		
	  // Build the Sig_UW segment
	    SigUWc=new covMat*[pu];	  
	    for (int ii=0; ii<pu; ii++) {
	      SigUWc[ii]=new covMat(xtheta,d->zt,pp->betaU[ii],gsl_vector_get(pp->lamUz,ii));
 	    }
        if (verbose) printf("Completed precomputation of SigUWc, it's size is %dx%d\n",
			    int(SigUWc[0]->cov->size1),int(SigUWc[0]->cov->size2));
   
	  // allocate the intermediate product W = SigUW(ii) * inv(SigW(ii))
	    W=new gsl_matrix*[pu];
	    for (int ii=0; ii<pu; ii++) {
	 	  W[ii]=gsl_matrix_alloc(n,m);
	    }
	    computeW();
	    if (verbose) printf("Completed precomputation of W, first float %f\n",gsl_matrix_get(W[0],0,0)); 

	  // allocate and precompute SigUgW (block diagonal elements) = SigU(ii) - W(ii)*SigUW(ii)'
	    SigUgW=new gsl_matrix*[pu];
	    for (int ii=0; ii<pu; ii++) SigUgW[ii]=gsl_matrix_alloc(n,n);
  	    computeSigUgW();
        if (verbose) printf("Completed precomputation of SigUgW\n");

	  // allocate the remaining matrices to be computed on demand. 
	    SigVUgWc=new covMat(n*(pv+pu));
	    MuVUgW=gsl_vector_alloc(n*(pv+pu));
	    SigVUgW_GP=new GPmodel(MuVUgW,SigVUgWc);
	  }

      if (verbose) printf("Completed allocation and precomputation\n");

  }
   ~LAmodel() {
   int verbose=0;
   if (n>0) if (verbose) printf("LAmodel Destructor: freeing xtheta\n");
   if (n>0) gsl_matrix_free(xtheta);
   if (verbose) printf("LAmodel Destructor: deleting covariances\n");
   if (n>0) delete SigVc;
   if (n>0) { for (int ii=0; ii<pu; ii++) delete SigUc[ii];  delete SigUc; }
   for (int ii=0; ii<pu; ii++) delete SigWc[ii];  delete SigWc;
   if (n>0) { for (int ii=0; ii<pu; ii++) delete SigUWc[ii]; delete SigUWc; }
   if (verbose) printf("LAmodel Destructor: deleting SigWGP\n");
   for (int ii=0; ii<pu; ii++) delete SigWGP[ii]; delete SigWGP;
   if (n>0) if (verbose) printf("LAmodel Destructor: freeing W\n");
   if (n>0) { for (int ii=0; ii<pu; ii++) gsl_matrix_free(W[ii]); delete W; }
   if (n>0) if (verbose) printf("LAmodel Destructor: deleting SigUgW\n");
   if (n>0) { for (int ii=0; ii<pu; ii++) gsl_matrix_free(SigUgW[ii]); delete SigUgW; }
   if (n>0) if (verbose) printf("LAmodel Destructor: deleting SigVUgWc\n");
   if (n>0) delete SigVUgWc;
   if (n>0) if (verbose) printf("LAmodel Destructor: freeing MuVUgW\n");
   if (n>0) gsl_vector_free(MuVUgW);
   if (n>0) if (verbose) printf("LAmodel Destructor: deleting SigVUgW_GP\n");
   if (n>0) delete SigVUgW_GP;
   if (verbose) printf("LAmodel Destructor: completed destructor\n");

   delete pp;
 }

  void computeW() {
    gsl_vector_view tviewx, tviewb;
	gsl_matrix *ch=gsl_matrix_alloc(m,m);
    for (int ii=0; ii<pu; ii++) {
	  gsl_matrix_memcpy(ch,SigWc[ii]->cov);
	  // get the cholesky to use for the solve(s)
      gsl_linalg_cholesky_decomp(ch);
	  // Solve for each row of the system  
	  for (int jj=0; jj<n; jj++) {
	    tviewb=gsl_matrix_row(SigUWc[ii]->cov,jj);
        tviewx=gsl_matrix_row(W[ii],jj);
  	    gsl_linalg_cholesky_solve(ch,&tviewb.vector,&tviewx.vector);
  	  }
	}
	gsl_matrix_free(ch);
  }  
  void computeSigUgW() {
	for (int ii=0; ii<pu; ii++) {
	  gsl_matrix_memcpy(SigUgW[ii],SigUc[ii]->cov);
	  gsl_blas_dgemm(CblasNoTrans,CblasTrans,-1.0,W[ii],SigUWc[ii]->cov,1.0,SigUgW[ii]);
    }
  }
  
  void updateTheta(int ind, double val) {
    //printf("Updating theta...\n");
    gsl_vector_set(pp->theta,ind,val);
	// update xtheta 
	for (int ii=0; ii<n; ii++) {
	  gsl_matrix_set(xtheta,ii,p+ind,val);
	}
	for (int ii=0; ii<pu; ii++) {
	  SigUc[ii]->computeDist(); SigUc[ii]->computeCov();
	  SigUWc[ii]->computeDist(); SigUWc[ii]->computeCov();
	}
	computeW();
	computeSigUgW();
  }
  void updateBetaV(int indi, int indj, double val) {
    gsl_vector_set(pp->betaV[indi],indj,val);
	//start back here
	SigVc->computeDist(); SigVc->computeCov();
  }
  void updateBetaU(int indi, int indj, double val) {
    gsl_vector_set(pp->betaU[indi],indj,val);
	if (n>0) { SigUc[indi]->computeDist(); SigUc[indi]->computeCov(); }
	SigWc[indi]->computeDist(); SigWc[indi]->computeCov();
	if (n>0) { SigUWc[indi]->computeDist(); SigUWc[indi]->computeCov(); }
	SigWGP[indi]->calcLogLik();
	if (n>0) computeW();
	if (n>0) computeSigUgW();
  }
  void updateLamVz(int ind, double val) {
    gsl_vector_set(pp->lamVz,ind,val);
	SigVc->precz=gsl_vector_get(pp->lamVz,ind);	SigVc->computeDist(); SigVc->computeCov();
  }
  void updateLamUz(int ind, double val) {
	gsl_vector_set(pp->lamUz,ind,val);
	if (n>0) {SigUc[ind]->precz=gsl_vector_get(pp->lamUz,ind); SigUc[ind]->computeDist(); SigUc[ind]->computeCov();}
	SigWc[ind]->precz=gsl_vector_get(pp->lamUz,ind); SigWc[ind]->computeDist(); SigWc[ind]->computeCov();
	if (n>0) {SigUWc[ind]->precz=gsl_vector_get(pp->lamUz,ind); SigUWc[ind]->computeDist(); SigUWc[ind]->computeCov();}
	SigWGP[ind]->calcLogLik();
	if (n>0) computeW();
	if (n>0) computeSigUgW();
  }
  void updateLamWs(int ind, double val) {
	gsl_vector_set(pp->lamWs,ind,val);
	if (n>0) {SigUc[ind]->precs=gsl_vector_get(pp->lamWs,ind); SigUc[ind]->computeDist(); SigUc[ind]->computeCov();}
	SigWc[ind]->precs =  1.0/(1.0/(gsl_vector_get(pp->LamSim,ind)*pp->lamWOs) + 1.0/gsl_vector_get(pp->lamWs,ind));
	SigWc[ind]->computeDist(); SigWc[ind]->computeCov();
	SigWGP[ind]->calcLogLik();
	if (n>0) computeW();
	if (n>0) computeSigUgW();
  }
  void updateLamWOs(double val) {
	pp->lamWOs=val;
	for (int ind=0; ind<pu; ind++) {
 	  SigWc[ind]->precs =  1.0/(1.0/(gsl_vector_get(pp->LamSim,ind)*pp->lamWOs) + 1.0/gsl_vector_get(pp->lamWs,ind));
	  SigWc[ind]->computeDist(); SigWc[ind]->computeCov();
	  SigWGP[ind]->calcLogLik();
	}
	if (n>0) computeW();
	if (n>0) computeSigUgW();
  }
  void updateLamOs(double val) {
     pp->lamOs=val;
  }
  
  double computeLogLik() {
	// logLik = logLik of W + logLik of VU|W

    // Log likelihood of the Ws are stored precomputed products
	// (because they may only need to be computed piecemeal depending on the parameter update)
    double logLikW=0;
	int verbose=0;

	if (verbose) printf("getting WGP likelihoods...\n");
	for (int ii=0; ii<pu; ii++) logLikW+=SigWGP[ii]->getLogLik();
    if (verbose) printf("completed, \n");

    if (n>0) {
	  // SigVUgW = 
	  //	| SigV		zeros |     + SigObs/lamOs
	  //	| zeros		SigUgW|    

      if (verbose) printf("first stage SigVUgWc ...");
      gsl_matrix_memcpy(SigVUgWc->cov,pp->SigObs);
	  gsl_matrix_scale(SigVUgWc->cov,1.0/pp->lamOs);
	  if (verbose) printf("completed\n");

	  gsl_matrix_view tviewm;
	  if (verbose) printf("second stage SigVUgWc ...");
	  for (int ii=0; ii<pv; ii++) {
	    tviewm=gsl_matrix_submatrix(SigVUgWc->cov,ii*n,ii*n,n,n);
	    //printf("tvm size=%dx%d\n",(&tviewm.matrix)->size1,(&tviewm.matrix)->size2);
	    //printf("SigVc size=%dx%d\n",SigVc->cov->size1,SigVc->cov->size2);	  
	    gsl_matrix_add(&tviewm.matrix,SigVc->cov);
	  }
	  if (verbose) printf("completed\n");
	  if (verbose) printf("third stage SigVUgWc ...");
	  for (int ii=0; ii<pu; ii++) {
	    tviewm=gsl_matrix_submatrix(SigVUgWc->cov,n*(pv+ii),n*(pv+ii),n,n);
	    gsl_matrix_add(&tviewm.matrix,SigUgW[ii]);
	  }
	  if (verbose) printf("completed\n");

      if (verbose) printf("constructing MuVUgW ...");
	  gsl_vector_view tviewv;
	  gsl_vector_memcpy(MuVUgW,d->vu);
	  for (int ii=0; ii<pu; ii++) {
	    tviewv=gsl_vector_subvector(MuVUgW,n*(pv+ii),n);
	    gsl_blas_dgemv(CblasNoTrans, -1.0, W[ii],d->w[ii],1.0,&tviewv.vector);
	  }
      if (verbose) printf("completed.\n");
    } 

	logLik=logLikW;
	if (n>0) logLik+=SigVUgW_GP->calcLogLik();

    return ( logLik) ;
  }
  
  double getLogLik() {return logLik;}

};

