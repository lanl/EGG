#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>

#include <cuda_runtime.h>

#include "cublas_v2.h"
#include "cusolverDn.h"
#include "helper_cuda.h"

#include "helper_cusolver.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <iostream>

//#define HOST_COVARIANCE
extern cublasHandle_t cuhandle;
extern cublasStatus_t cstatus;

struct cal_log_lik : public thrust::unary_function<int, void>
{
  int n_;
  double *x_, *A_, *result_;

  cal_log_lik(int n, double* x, double* A, double *result) :
                          n_(n), x_(x), A_(A), result_(result) {}

  __host__ __device__
  void operator()(int j)
  {
	result_[j] = - 0.5*x_[j]*x_[j]-log(A_[j + n_*j]);	
  }
};


int factorCHOLs(
    cusolverDnHandle_t handle,
    int n,
    double *A, double *x, double *h_loglik)
{
   thrust::device_vector<double> h_vec(n);
  /* checkCudaErrors(cudaMemcpy(thrust::raw_pointer_cast(&*h_vec.begin()),(double *)x, sizeof(double)*n, cudaMemcpyDeviceToHost));
        printf("\nBefore solving\n");
    for (int ii=0; ii<n; ii++)
    {
        std::cout<<h_vec[ii]<<" ";
    } 
  getchar();*/
    int bufferSize = 0;
    int *info = NULL;
    double *buffer = NULL;
    //int h_info = 0;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
    //double *x = NULL;

    checkCudaErrors(cusolverDnDpotrf_bufferSize(handle, uplo, n, (double*)A, n, &bufferSize));
    //fprintf(stderr, "Buffer size %d\n",bufferSize);
    
    checkCudaErrors(cudaMalloc(&info, sizeof(int)));
    checkCudaErrors(cudaMalloc(&buffer, sizeof(double)*bufferSize));
    checkCudaErrors(cudaMemset(info, 0, sizeof(int)));
    //checkCudaErrors(cudaMalloc(x, sizeof(double)*n));

///*
    //double start, stop;
    //double time_solve;
    //fprintf(stdout, "Starting Cholesky\n");
    //start = second();
//*/
    checkCudaErrors(cusolverDnDpotrf(handle, uplo, n, A, n, buffer, bufferSize, info));
//	checkCudaErrors(cudaDeviceSynchronize());
   //stop = second();
    //time_solve = stop - start;
    //fprintf (stdout, "Timing: cholesky = %10.6f sec\n", time_solve);

/*
    start = second();
    checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));
    stop = second();
    time_solve = stop - start;
    fprintf (stdout, "Timing: cudaMemcpy = %10.6f sec\n", time_solve);
*/


    /*
    if ( 0 != h_info )
        fprintf(stderr, "Error: Cholesky factorization failed with Error code %d\n",h_info);
    */
	//checkCudaErrors(cudaDeviceSynchronize());
//	checkCudaErrors(cudaMemcpy(x, b, sizeof(double)*n, cudaMemcpyDeviceToDevice));
//    checkCudaErrors(cusolverDnDpotrs(handle, uplo, n, 1, A, n, x, n, info));
	// Use cublas : cublasDtrsv
//	cublasStatus_t cublasDtrsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double *A, int lda, double *x, int incx)

//   cublasHandle_t cuhandle;
//   cublasStatus_t cstatus;
   /* Initialize CUBLAS */
    //printf("simpleCUBLAS test running..\n");

/*    cstatus = cublasCreate(&cuhandle);

    if (cstatus != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
        return EXIT_FAILURE;
    }
*/ 
   //uplo = CUBLAS_FILL_MODE_LOWER;
   cublasDiagType_t diag = CUBLAS_DIAG_NON_UNIT;
   cublasOperation_t trans = CUBLAS_OP_T;
  
//   start = second();  

    cstatus = cublasDtrsv( cuhandle, uplo, trans, diag, n, A, n, x, 1);
/*   
    stop = second();
    time_solve = stop - start;
    fprintf (stdout, "Timing: cublasDtrsv = %10.6f sec\n", time_solve);
*/
  // std::cout<<"cublas status is "<<cstatus<<std::endl; 
   // checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

   // if ( 0 != h_info )
   //     fprintf(stderr, "Error: Cholesky factorization failed with Error code %d\n",h_info);
    

//	checkCudaErrors(cudaDeviceSynchronize());
 

//   checkCudaErrors(cudaMemcpy(thrust::raw_pointer_cast(&*h_vec.begin()), x, sizeof(double)*n, cudaMemcpyDeviceToHost));
   /*     printf("After solving\n");
    for (int ii=0; ii<n; ii++)
    {
        std::cout<<h_vec[ii]<<" ";
    }
    getchar();
   */
///*
  
/*  stop = second();
    time_solve = stop - start;
    fprintf (stdout, "Timing: cholesky = %10.6f sec\n", time_solve);
*/
    thrust::device_vector<double> t_result(n);
    thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(n),
                     cal_log_lik(n, x, A, thrust::raw_pointer_cast(&*t_result.begin()) ));
    (*h_loglik) = thrust::reduce(t_result.begin(), t_result.end(), (double) 0, thrust::plus<double>());

    //std::cout<<"cal loglik is "<<sum<<std::endl;
    //getchar();

/*
    //checkCudaErrors(cudaMemcpy((*tvec), x, sizeof(double)*n, cudaMemcpyDeviceToHost));
    thrust::device_vector<double> loglik(1,0.0);
    //double loglik=0.0;
    for (int ii=0; ii<n; ii++)
    {
	//printf("%lf %lf",A[ii + n*ii]);// - 0.5*x[ii]*x[ii];
	loglik[0]-=- 0.5*x[ii]*x[ii]-log(A[ii + n*ii]);// - 0.5*x[ii]*x[ii];
    }
*/
  //*h_loglik = loglik[0];
    //fprintf (stdout, "LogLik is = %10.6f sec\n", loglik[0]);
	//checkCudaErrors(cudaMemcpy(h_loglik, thrust::raw_pointer_cast(&*loglik.begin()), sizeof(double), cudaMemcpyDeviceToHost));

    if (info  ) { checkCudaErrors(cudaFree(info)); }
    if (buffer) { checkCudaErrors(cudaFree(buffer)); }
    
    /* Shutdown */
   // cstatus = cublasDestroy(cuhandle);
   
    return 0;
}


int factorCHOL(
    cusolverDnHandle_t handle,
    int n,
    double *A, double **Cmat)
{
    int bufferSize = 0;
    int *info = NULL;
    double *buffer = NULL;
    int h_info = 0;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

    checkCudaErrors(cusolverDnDpotrf_bufferSize(handle, uplo, n, (double*)A, n, &bufferSize));
    //fprintf(stderr, "Buffer size %d\n",bufferSize);
    
    checkCudaErrors(cudaMalloc(&info, sizeof(int)));
    checkCudaErrors(cudaMalloc(&buffer, sizeof(double)*bufferSize));
    checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

//*
    //double start, stop;
    //double time_solve;
    fprintf(stdout, "Starting Cholesky\n");
    //start = second();
//*/
    checkCudaErrors(cusolverDnDpotrf(handle, uplo, n, A, n, buffer, bufferSize, info));
    checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    /*
    if ( 0 != h_info )
        fprintf(stderr, "Error: Cholesky factorization failed with Error code %d\n",h_info);
    */
    checkCudaErrors(cudaDeviceSynchronize());
//*
    //stop = second();
    //time_solve = stop - start;
    //fprintf (stdout, "Timing: cholesky = %10.6f sec\n", time_solve);
//*/
    checkCudaErrors(cudaMemcpy((*Cmat), A, sizeof(double)*n*n, cudaMemcpyDeviceToHost));

    if (info  ) { checkCudaErrors(cudaFree(info)); }
    if (buffer) { checkCudaErrors(cudaFree(buffer)); }

    return 0;
}


struct build_covariance_matrix : public thrust::unary_function<int, void>
{
  int n_;
  double lam_, b_;
  double *x_, *A_;

  build_covariance_matrix(int n, double lam, double b, double* x, double* A) :
                          n_(n), lam_(lam), b_(b), x_(x), A_(A) {}

  __host__ __device__
  void operator()(int j)
  {
      A_[j + j*n_] = 1.0 + lam_;
      for (int k=0; k<j; k++) {
        double cov = b_*exp(-(x_[j]-x_[k])*(x_[j]-x_[k]));
        A_[j + k*n_] = cov;
        A_[k + j*n_] = cov;
      }

  }
};
