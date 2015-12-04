/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include <math.h>
#include "support.h"
#include "kernel.cu"

int main(int argc, char**argv) {

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    int THREADS_PER_BLOCK = 128;
    unsigned int n;
    if(argc == 1) {
        n = 10000;
    } else if(argc == 2) {
        n = atoi(argv[1]);
    }  else if(argc == 3) {
        n = atoi(argv[1]);
	THREADS_PER_BLOCK = atoi(argv[2]);
    } else {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./vecadd               # Vector of size 10,000 is used"
           "\n    Usage: ./vecadd <m>           # Vector of size m is used"
           "\n");
        exit(0);
    }

    float* A_h = (float*) malloc( sizeof(float)*n );
    for (unsigned int i=0; i < n; i++) { A_h[i] = (rand()%100)/100.00; }

    float* B_h = (float*) malloc( sizeof(float)*n );
    for (unsigned int i=0; i < n; i++) { B_h[i] = (rand()%100)/100.00; }

    float* C_h = (float*) malloc( sizeof(float)*n );

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    Vector size = %u\n", n);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    float* A_d ; 
    int success = cudaMalloc( (void**) &A_d, sizeof(float)*n) ; 

    float* B_d ; 
    success = cudaMalloc( (void**) &B_d, sizeof(float)*n) ; 


    float* C_d ; 
    success = cudaMalloc( (void**) &C_d, sizeof(float)*n) ; 

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    success = cudaMemcpy (A_d,A_h,sizeof(float)*n, cudaMemcpyHostToDevice);

    success = cudaMemcpy (B_d,B_h,sizeof(float)*n, cudaMemcpyHostToDevice);


    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel ----------------------------------------------------------

    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);

/*
    int numBlocks = ceil((n-1)/256) + 1;
    printf("NumBlocks %f\n",numBlocks) ;

    vecAddKernel<<<numBlocks,256>>>(A_d,B_d,C_d,n) ;
*/
	dim3 DimGrid(ceil(1.0*n/THREADS_PER_BLOCK),1,1) ;
	dim3 DimBlock(THREADS_PER_BLOCK,1,1) ; 

   printf("DimGrid: %dX%dX%d \nDimBlock: %dX%dX%d\n", DimGrid.x, DimGrid.y, DimGrid.z, DimBlock.x, DimBlock.y, DimBlock.z ); fflush(stdout);

    vecAddKernel<<<DimGrid,DimBlock>>>(A_d,B_d,C_d,n) ;
    cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    success = cudaMemcpy (C_h,C_d,sizeof(float)*n, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify(A_h, B_h, C_h, n);

    // Free memory ------------------------------------------------------------

    free(A_h);
    free(B_h);
    free(C_h);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return 0;

}

