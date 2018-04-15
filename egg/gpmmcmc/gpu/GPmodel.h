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

#include "cuChol.h"


extern cusolverDnHandle_t handle;
extern cudaStream_t stream;
extern int GPU;
extern int chol_cnt;

struct compute_cov_likelihood : public thrust::unary_function<int, void>
{
  int n_,p_,precsPresent_;
  double precz_,precs_;
  double *x_, *A_, *lscale_;

  compute_cov_likelihood(int n,int p, double precz,int precsPresent, double precs, double* x, double* A, double* lscale ) :
              n_(n), p_(p), precz_(precz),precsPresent_(precsPresent),precs_(precs), x_(x), A_(A), lscale_(lscale) {}

  __host__ __device__
  void operator()(long j)
  {
        int ii = n_ - 2 - floor(sqrt((double)(-8*j + 4*n_*(n_-1)-7))/2.0 - 0.5);
        int jj = j + ii + 1 - n_*(n_-1)/2 + (n_-ii)*((n_-ii)-1)/2;
	double td2=0;
      	for (int k=0; k<p_; k++) 
	{
        	double dist = (x_[ii*p_+ k]-x_[jj*p_+ k]); // * (x_[ii*p_+k]-x_[jj*p_+ k]);
		td2 += dist*dist * lscale_[k];
      	}  
        td2 = exp(-1.0*td2) / precz_;
        A_[ii + n_*jj]=td2;
        A_[ii*n_ + jj]=td2;
	/*
 	if(jj==ii+1)
	{
		if (precsPresent_) A_[ii + n_*ii]=(1.0/precz_ + 1.0/precs_);
		else A_[ii + n_*ii] = (1.0/precz_); 
	}*/
  }
};

struct compute_cov_matrix : public thrust::unary_function<int, void>
{
  int n_,p_,precsPresent_;
  double precz_,precs_;
  double *x_, *A_, *lscale_;

  compute_cov_matrix(int n,int p, double precz,int precsPresent, double precs, double* x, double* A, double* lscale ) :
              n_(n), p_(p), precz_(precz),precsPresent_(precsPresent),precs_(precs), x_(x), A_(A), lscale_(lscale) {}

  __host__ __device__
  void operator()(long j)
  {
        int ii = n_ - 2 - floor(sqrt((double)(-8*j + 4*n_*(n_-1)-7))/2.0 - 0.5);
        int jj = j + ii + 1 - n_*(n_-1)/2 + (n_-ii)*((n_-ii)-1)/2;
	double td2=0;
      	for (int k=0; k<p_; k++) 
	{
        	double dist = (x_[ii*p_+ k]-x_[jj*p_+ k]); // * (x_[ii*p_+k]-x_[jj*p_+ k]);
		td2 += dist*dist * lscale_[k];
      	}  
        td2 = exp(-1.0*td2) / precz_;
        A_[ii + n_*jj]=td2;
        A_[ii*n_ + jj]=td2;
	/*
 	if(jj==ii+1)
	{
		if (precsPresent_) A_[ii + n_*ii]=(1.0/precz_ + 1.0/precs_);
		else A_[ii + n_*ii] = (1.0/precz_); 
	}*/
  }
};

struct compute_dist_cov_matrix : public thrust::unary_function<int, void>
{
  int n_,p_;
  double *x_, *A_;

  compute_dist_cov_matrix(int n,int p, double* x, double* A) :
                          n_(n), p_(p), x_(x), A_(A) {}

  __host__ __device__
  void operator()(long j)
  {
	int ii = n_ - 2 - floor(sqrt((double)(-8*j + 4*n_*(n_-1)-7))/2.0 - 0.5);
	int jj = j + ii + 1 - n_*(n_-1)/2 + (n_-ii)*((n_-ii)-1)/2;
      for (int k=0; k<p_; k++) {
        double dist = (x_[ii*p_+ k]-x_[jj*p_+ k]); // * (x_[ii*p_+k]-x_[jj*p_+ k]);
        A_[j*p_+k] = dist*dist;
      }

  }
};


struct compute_scaled_dist_cov_matrix : public thrust::unary_function<int, void>
{
  int n_,p_;
  double precz_;
  double *x_, *A_,*lscale_;

  compute_scaled_dist_cov_matrix(int n,int p, double precz, double* x, double* A, double* lscale) :
                          n_(n), p_(p), precz_(precz), x_(x), A_(A), lscale_(lscale) {}

  __host__ __device__
  void operator()(long j)
  {
	int ii = n_ - 2 - floor(sqrt((double)(-8*j + 4*n_*(n_-1)-7))/2.0 - 0.5);
	int jj = j + ii + 1 - n_*(n_-1)/2 + (n_-ii)*((n_-ii)-1)/2;
	double td2=0;
	for (int k=0; k<p_; k++) 
	{
        	td2 += x_[j*p_+k] * lscale_[k];
      	}
	td2 = exp(-1.0*td2) / precz_;
	A_[ii + n_*jj]=td2;
	A_[ii*n_ + jj]=td2;
  }
};


struct compute_scaling_on_cov_matrix1 : public thrust::unary_function<int, void>
{
  int n_,p_;
  double *A_;
  double precz_,precs_;

  compute_scaling_on_cov_matrix1(int n, double precz, double precs, double* A) :
                          n_(n), precz_(precz), precs_(precs), A_(A) {}

  __host__ __device__
  void operator()(int j)
  {
        A_[j*n_+j] = 1.0/precz_+ 1.0/precs_;
  }
};

struct compute_scaling_on_cov_matrix2 : public thrust::unary_function<int, void>
{
  int n_;
  double *A_;
  double precz_;

  compute_scaling_on_cov_matrix2(int n, double precz, double* A) :
                          n_(n), precz_(precz), A_(A) {}

  __host__ __device__
  void operator()(int j)
  {
        A_[j*n_+j] = 1.0/precz_;
  }
};


void applyCuChols(gsl_matrix *ch, gsl_vector * tvec, int n, double *loglik)
{
    /*
    printf("in applyCHol\n");
    for (int ii=0; ii<n; ii++)
    {
        printf("%lf ",gsl_vector_get(tvec,ii));
    }
    */
    thrust::device_vector<double> t_A, t_x;
    //thrust::host_vector<double> h_A, h_x;

    t_A.resize(n*n);
    t_x.resize(n);
    //h_x.resize(n);
    //double start, stop;
    //double time_solve;
    //int timer=clock();
    //start = second();
    checkCudaErrors(cudaMemcpy(thrust::raw_pointer_cast(&*t_A.begin()),(double *)ch->data,sizeof(double)*n*n,cudaMemcpyHostToDevice));
    //printf("before chChol completed in %f\n",1.0*(clock()-timer)/CLOCKS_PER_SEC);
    checkCudaErrors(cudaMemcpy(thrust::raw_pointer_cast(&*t_x.begin()),(double *)tvec->data, sizeof(double)*n, cudaMemcpyHostToDevice));
    //printf("before chChol completed in %f\n",1.0*(clock()-timer)/CLOCKS_PER_SEC);
    //stop = second();
    //time_solve = stop - start;
    //fprintf (stdout, "Timing: send cudamemcpy = %10.6f sec\n", time_solve);
	//thrust::copy(t_x.begin(),t_x.end(),h_x.begin());
    /*printf("\nin2 applyCHol\n");
    for (int ii=0; ii<n; ii++)
    {
        printf("%lf ",h_x[ii]);
    }
	getchar();
    */
	//timer=clock();
    //factorCHOL(handle, n, thrust::raw_pointer_cast(&*t_A.begin()),&(ch->data));
    factorCHOLs(handle, n, thrust::raw_pointer_cast(&*t_A.begin()),thrust::raw_pointer_cast(&*t_x.begin()),loglik);
	//printf("chChol completed in %f\n",1.0*(clock()-timer)/CLOCKS_PER_SEC);
}


void applyCuChol(gsl_matrix *ch, int n)
{

    thrust::device_vector<double> t_A;

    t_A.resize(n*n);

    //int timer=clock();
    cudaMemcpy(thrust::raw_pointer_cast(&*t_A.begin()),(double *)ch->data,sizeof(double)*n*n,cudaMemcpyHostToDevice);
    //printf("before chChol completed in %f\n",1.0*(clock()-timer)/CLOCKS_PER_SEC);

	//timer=clock();
    factorCHOL(handle, n, thrust::raw_pointer_cast(&*t_A.begin()),&(ch->data));
	//printf("chChol completed in %f\n",1.0*(clock()-timer)/CLOCKS_PER_SEC);
}

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
    covMat(covMat *matIn,gsl_matrix *xIn, gsl_vector *lscaleIn) { // copy another covMat
           d2Dealloc=0;//matIn->d2Dealloc;
           type=matIn->type;
           precsPresent=matIn->precsPresent;
           n=matIn->n;
           m=matIn->m;
           p=matIn->p;
           precs=matIn->precs;
           precz=matIn->precz;
	   cov=gsl_matrix_calloc(n,n);
           gsl_matrix_memcpy(cov,matIn->cov);
	   //x=gsl_matrix_calloc(matIn->x->size1,matIn->x->size2);
           //gsl_matrix_memcpy(x,matIn->x);
           x = xIn;
	   //z=gsl_matrix_calloc(matIn->z->size1,matIn->z->size2);
           //gsl_matrix_memcpy(z,matIn->z);
	   //lscale = gsl_vector_calloc(matIn->lscale->size);
	   //gsl_vector_memcpy(lscale,matIn->lscale);
	   lscale=lscaleIn;
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
    void computeGPUDistCovLikelihood(gsl_vector *htvec, double *loglik)
    {
	thrust::device_vector<double> t_A, t_x, t_lscale;
        int up_tr_size = n*(n-1)/2;
        t_A.resize(n*n);
        t_lscale.resize(p);
        t_x.resize(n*p);
        cudaMemcpy(thrust::raw_pointer_cast(&*t_x.begin()),(double *)x->data,sizeof(double)*n*p,cudaMemcpyHostToDevice);
        cudaMemcpy(thrust::raw_pointer_cast(&*t_lscale.begin()),(double *)lscale->data,sizeof(double)*p,cudaMemcpyHostToDevice);
        thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(up_tr_size),
                     compute_cov_matrix(n, p, precz, precsPresent, precs, thrust::raw_pointer_cast(&*t_x.begin()),
                                             thrust::raw_pointer_cast(&*t_A.begin()), thrust::raw_pointer_cast(&*t_lscale.begin()) ));
    
	if(precsPresent)
        {
                thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(n),
                     compute_scaling_on_cov_matrix1(n, precz, precs, thrust::raw_pointer_cast(&*t_A.begin()) ));
        }
        else
        {
                thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(n),
                     compute_scaling_on_cov_matrix2(n, precz, thrust::raw_pointer_cast(&*t_A.begin()) ));
        }
        //checkCudaErrors(cudaMemcpy((double *)cov->data, thrust::raw_pointer_cast(&*t_A.begin()), sizeof(double)*n*n, cudaMemcpyDeviceToHost));
	// Now compute likelihood from this matrix
    //thrust::host_vector<double> h_A, h_x;

    t_x.resize(n);
    //h_x.resize(n);

    //int timer=clock();
    //printf("before chChol completed in %f\n",1.0*(clock()-timer)/CLOCKS_PER_SEC);
    checkCudaErrors(cudaMemcpy(thrust::raw_pointer_cast(&*t_x.begin()),(double *)htvec->data, sizeof(double)*n, cudaMemcpyHostToDevice));
	//thrust::copy(t_x.begin(),t_x.end(),h_x.begin());
    /*printf("\nin2 applyCHol\n");
    for (int ii=0; ii<n; ii++)
    {
        printf("%lf ",h_x[ii]);
    }
	getchar();
    */
	//timer=clock();
    //factorCHOL(handle, n, thrust::raw_pointer_cast(&*t_A.begin()),&(ch->data));
    factorCHOLs(handle, n, thrust::raw_pointer_cast(&*t_A.begin()),thrust::raw_pointer_cast(&*t_x.begin()),loglik);
	//printf("chChol completed in %f\n",1.0*(clock()-timer)/CLOCKS_PER_SEC);
    chol_cnt++;
    }


    void computeGPUDistCov()
    {
	thrust::device_vector<double> t_A, t_x, t_lscale;
        int up_tr_size = n*(n-1)/2;
        t_A.resize(n*n);
        t_lscale.resize(p);
        t_x.resize(n*p);
	cudaMemcpy(thrust::raw_pointer_cast(&*t_x.begin()),(double *)x->data,sizeof(double)*n*p,cudaMemcpyHostToDevice);
        cudaMemcpy(thrust::raw_pointer_cast(&*t_lscale.begin()),(double *)lscale->data,sizeof(double)*p,cudaMemcpyHostToDevice);
        thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(up_tr_size),
                     compute_cov_matrix(n, p, precz, precsPresent, precs, thrust::raw_pointer_cast(&*t_x.begin()),
                                             thrust::raw_pointer_cast(&*t_A.begin()), thrust::raw_pointer_cast(&*t_lscale.begin()) ));
        if(precsPresent)
	{
		thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(n),
                     compute_scaling_on_cov_matrix1(n, precz, precs, thrust::raw_pointer_cast(&*t_A.begin()) ));
	}
	else
	{
		thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(n),
                     compute_scaling_on_cov_matrix2(n, precz, thrust::raw_pointer_cast(&*t_A.begin()) ));
	}
        checkCudaErrors(cudaMemcpy((double *)cov->data, thrust::raw_pointer_cast(&*t_A.begin()), sizeof(double)*n*n, cudaMemcpyDeviceToHost));
	/*for (int ii=0; ii<n; ii++) {
               if (precsPresent) gsl_matrix_set(cov,ii,ii,1.0/precz + 1.0/precs);
                            else gsl_matrix_set(cov,ii,ii,1.0/precz);
             }
	*/
    }
    void computeDist() {

	   int dind=0;
	   double tdub;
	   if (type==1) {

	//printf("Here compute Dist\n");


	if(GPU==0)
	{
	//printf("Type 1 ");
	     for (int ii=0; ii<n-1; ii++)
	       for (int jj=ii+1; jj<n; jj++)
		     for (int kk=0; kk<p; kk++) {
		         tdub=gsl_matrix_get(x,ii,kk) - gsl_matrix_get(x,jj,kk);
			     d2[dind++]=tdub*tdub;
	     }
	//printf(" dims %d %d \n",x->size1, x->size2);
	//printf(" n=%d p= %d \n",n, p);
	//getchar();
	}
	else{
	// check output for CUDA calls
	thrust::device_vector<double> t_A, t_x;
	int up_tr_size = n*(n-1)/2;
	t_A.resize(up_tr_size*p);
	t_x.resize(n*p);
	//t_A[0]=-10;
	//thrust::copy(x->data, x->data+n*n, t_x.begin());
	cudaMemcpy(thrust::raw_pointer_cast(&*t_x.begin()),(double *)x->data,sizeof(double)*n*p,cudaMemcpyHostToDevice);
	thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(up_tr_size),
                     compute_dist_cov_matrix(n, p, thrust::raw_pointer_cast(&*t_x.begin()),
                                             thrust::raw_pointer_cast(&*t_A.begin())));
	/*double test;
	thrust::copy(t_A.begin(), t_A.begin()+1, &test);
	fprintf(stdout, "Force result: %lf Serial d2[] %lf \n", test,d2[0]);
	double* h_A = (double*)malloc(sizeof(double)*up_tr_size*p);
	checkCudaErrors(cudaMemcpy(h_A, thrust::raw_pointer_cast(&*t_A.begin()), sizeof(double)*up_tr_size*p, cudaMemcpyDeviceToHost));
	double err=0.0;
	for(int i =0; i<up_tr_size*p; i++)
	{
		//compute differences
		err+=fabs(d2[i]-h_A[i]);
	}

	printf(" Here outside: Error is %lf\n",err);
	*/	
	checkCudaErrors(cudaMemcpy(d2, thrust::raw_pointer_cast(&*t_A.begin()), sizeof(double)*up_tr_size*p, cudaMemcpyDeviceToHost));
		}
	//getchar();
	   }
	   if (type==2) {
	printf("Type 2 ");
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
	//printf("Here compute cov\n");

	if(GPU==0)
	{ 
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
	}
	else
	{
	//GPU computation starts
	//printf("lscale size is %d\n",lscale->size);
	thrust::device_vector<double> t_A, t_x, t_lscale;
        int up_tr_size = n*(n-1)/2;
	//printf("up_tr_size = %d, p= %d",up_tr_size,p);
        t_A.resize(n*n);
        t_lscale.resize(p);
        t_x.resize(up_tr_size*p);
	cudaMemcpy(thrust::raw_pointer_cast(&*t_x.begin()), (double *)d2, sizeof(double)*up_tr_size*p, cudaMemcpyHostToDevice);
	cudaMemcpy(thrust::raw_pointer_cast(&*t_lscale.begin()),(double *)lscale->data,sizeof(double)*p,cudaMemcpyHostToDevice);
        thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(up_tr_size),
                     compute_scaled_dist_cov_matrix(n, p, precz, thrust::raw_pointer_cast(&*t_x.begin()),
                                             thrust::raw_pointer_cast(&*t_A.begin()), thrust::raw_pointer_cast(&*t_lscale.begin()) ));
        checkCudaErrors(cudaMemcpy((double *)cov->data, thrust::raw_pointer_cast(&*t_A.begin()), sizeof(double)*n*n, cudaMemcpyDeviceToHost));
	}
	/*
	//check for errors between serial and parallel version
	double err = 0.0;
	for (int ii=0;ii<(n-1); ii++) {
               for (int jj=ii+1; jj<n; jj++) {
                     td2=0;
                     for (int kk=0; kk<p; kk++)
                           td2+=d2[dind++] * gsl_vector_get(lscale,kk);
                     td2=exp(-1.0*td2) / precz;
                     err+= fabs(gsl_matrix_get(cov,ii,jj)-td2);
                   }
             }
        printf(" Here outside: Error is %lf\n",err);
	getchar();
	*/
	/*double* h_A = (double*)malloc(sizeof(double)*n*n);
        checkCudaErrors(cudaMemcpy(h_A, thrust::raw_pointer_cast(&*t_A.begin()), sizeof(double)*n*n, cudaMemcpyDeviceToHost));
        double err=0.0;
        for(int i =0; i<n; i++)
        for(int j =0; j<n; j++)
        {
                //compute differences
               err+=fabs(gsl_matrix_get(cov,i,j)-h_A[i*n+j]);
        }*/
        //printf(" Here outside: Error is %lf\n",err);
	//getchar();
	//GPU ends
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
	gsl_vector *tvec,*tvec1;
	gsl_matrix *ch;//,*ch1;
	int n;
	int verbose;

  ~GPmodel() {
	gsl_vector_free(tvec);
	gsl_vector_free(tvec1);
	gsl_matrix_free(ch);
	//gsl_matrix_free(ch1);
  }
  GPmodel(gsl_vector *respIn, covMat *covIn) { // data-based constructor
	resp=gsl_vector_alloc(respIn->size);
	gsl_vector_memcpy(resp,respIn);
	//resp=respIn;
	cov=covIn;
	//cov=gsl_matrix_alloc(cov->n,cov->n);
	n=cov->n;
	verbose=0;
	tvec=gsl_vector_alloc(n);
	gsl_vector_memcpy(tvec,resp);
	tvec1=gsl_vector_alloc(n);
	ch=gsl_matrix_alloc(cov->n,cov->n);
	//ch1=gsl_matrix_alloc(cov->n,cov->n);
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
	//gsl_matrix_memcpy(ch1,cov->cov);

    if (verbose) 
    {
      printf("\ncov matrix:\n"); 
  	  for (int ii=0; ii<n; ii++) {
	    for (int jj=0; jj<n; jj++) {
	      printf("%8.5f  ",gsl_matrix_get(ch,ii,jj)); 
        }
	    printf("\n");
	  }
    }
	gsl_vector_memcpy(tvec,resp);
	//gsl_vector_memcpy(tvec1,resp);

// Lets take some timing here
/*
	printf("GPmodel: entered calcLogLok\n");
	int timer=clock();
	applyCuChol(ch1,n);
	printf("GPU Chol completed in %f\n",1.0*(clock()-timer)/CLOCKS_PER_SEC);

	timer=clock();
	gsl_linalg_cholesky_decomp(ch);
	printf("CPU Chol completed in %f\n",1.0*(clock()-timer)/CLOCKS_PER_SEC);
	getchar();
*/
/*
 	gsl_linalg_cholesky_decomp(ch1);
	double error = 0.0; 
	//printf("\nPrnting error :\n");
          for (int ii=0; ii<n; ii++) {
            for (int jj=0; jj<n; jj++) {
              error+=fabs(gsl_matrix_get(ch,ii,jj)-gsl_matrix_get(ch1,ii,jj));
              //printf("%8.5f  ",gsl_matrix_get(ch,ii,jj)-gsl_matrix_get(ch1,ii,jj));
        }
            //printf("\n");
          }
	printf(" Error = %lf ",error);
	//getchar();
*/	
	if(0)
	{

	if(GPU)	applyCuChol(ch,n);
	else gsl_linalg_cholesky_decomp(ch);
    
    if (verbose) 
    {
      printf("\ncholesky matrix:\n"); 
  	  for (int ii=0; ii<n; ii++) {
	    for (int jj=0; jj<n; jj++) {
	      printf("%8.5f  ",gsl_matrix_get(ch,ii,jj)); 
        }
	    printf("\n");
	  }
    }
//	getchar();	
	
	double logDet=0;
	for (int ii=0;ii<n; ii++)
	  logDet+=log(gsl_matrix_get(ch,ii,ii));
	
    if (verbose) printf("\nlog det: %8.5f \n",logDet);
	

    // blas routine for solving w/triangular matrix
	//gsl_blas_dtrsv(CblasUpper,CblasNoTrans,CblasNonUnit,ch,tvec);
	gsl_blas_dtrsv(CblasLower,CblasNoTrans,CblasNonUnit,ch,tvec);

    if (verbose) {
      printf("\nresult vector: \n");
  	  for (int ii=0; ii<n; ii++)
	      printf("%8.5f  ",gsl_vector_get(tvec,ii)); 
      printf("\n");
	}

	if (0) {
      printf("\nresult vector: \n");
          for (int ii=0; ii<n; ii++)
              printf("%8.5f  ",gsl_vector_get(tvec,ii));
      printf("\n");
        }

	logLik=-logDet;
	for (int ii=0; ii<cov->n; ii++)
	    logLik-=0.5*gsl_vector_get(tvec,ii)*gsl_vector_get(tvec,ii);
	//printf("correct loglik %f\n",logLik);
	}
	else
	{	
	double gpu_log=0.0;
	//int timer=clock();
	//double start, stop;
	//double time_solve;
    //    start = second();
	applyCuChols(ch,tvec,n, &gpu_log);
	chol_cnt++;
	//printf("GPU Chol completed in %f\n",1.0*(clock()-timer)/CLOCKS_PER_SEC);
	//stop = second();
	//time_solve = stop - start;
	//fprintf (stdout, "Timing: send total cholesky = %10.6f sec\n", time_solve);
	logLik = gpu_log;
	}
	//printf("loglik %f\n",logLik);
	//printf("correct loglik %f, gpu loglik %f\n",logLik,gpu_log);
	//getchar();
    return logLik;
  }
  void predict(covMat *cPred) {}
};
