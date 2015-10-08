/***
*	Ashutosh Dhar
*	Department of Electrical and Computer Engineeing
*	University of Illinois, Urbana-Champaign
*	
*/

#include <cuda.h>
#include <iostream>
#include <cstdio>
#include "support.h"

#define THREADS_PER_SM 2048
#define BLOCKS_PER_SM 32

using namespace std;

// Vector addition Kernel
// Device kernel

__global__ void bandwidthTest(const float *A, float *C, int numElements, int n_thread)
{
        long unsigned tid = blockDim.x * blockIdx.x + (threadIdx.x);
	int i;

        if (tid < numElements)
        {
			for(i=0; i<n_thread; i++)
	                        C[tid + (i * THREADS_PER_SM * BLOCKS_PER_SM)] = 2*A[tid + (i * THREADS_PER_SM * BLOCKS_PER_SM)];
        }

}

int main(int argc, char **argv)
{

	long unsigned n_elements_thread;
	long unsigned n_threads;
	long unsigned n_elements;
        long unsigned n_threads_per_block = THREADS_PER_SM/BLOCKS_PER_SM; //2048 threads per SM, 16 blocks per SM
	unsigned n_test = 5;
	
	Timer t_timer;

	if(argc > 1)
	{
		n_elements_thread = atoi(argv[1]);
	}
	else
	{
		n_elements_thread = 1;
	}
	
	if(argc > 2)
	{
		n_test = atoi(argv[2]);
	}

	n_elements = n_elements_thread * THREADS_PER_SM * BLOCKS_PER_SM;
	n_threads = n_elements / n_elements_thread;
	
	printf("Num elements in generated vector: %lu\n", n_elements);
	printf("Threads Per Block: %lu\n", n_threads_per_block);	
	
    	//Error Flag returned by cuda
    	cudaError_t errorFlag = cudaSuccess;
//	cudaError_t err = cudaSuccess;

    	//Set element sizes and allocate data sizes
    	int numElements = n_elements;
    	size_t sizeVector = (numElements * sizeof(float));

    	// Allocate the host input vector A
    	float *h_A = (float *)malloc(sizeVector);
    	// Allocate the host input vector B
//    	float *h_B = (float *)malloc(sizeVector);
    	// Allocate the host output vector C
    	float *h_C = (float *)malloc(sizeVector);

// From NVIDIA SDK, randomized initialization of data

    	// Verify that allocations succeeded
    	if (h_A == NULL || h_C == NULL)
    	{
        	fprintf(stderr, "Failed to allocate host vectors!\n");
        	exit(EXIT_FAILURE);
    	}

    	// Initialize the host input vectors
    	for (int i = 0; i < numElements; ++i)
    	{
        	h_A[i] = rand()/(float)RAND_MAX;
//        	h_B[i] = rand()/(float)RAND_MAX;
    	}

    	// Allocate the device input vector A
    	float *d_A = NULL;
    	errorFlag = cudaMalloc((void **)&d_A, sizeVector);

    	if (errorFlag != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(errorFlag));
        	exit(EXIT_FAILURE);
    	}

/*
    	// Allocate the device input vector B
    	float *d_B = NULL;
    	errorFlag = cudaMalloc((void **)&d_B, sizeVector);

    	if (errorFlag != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(errorFlag));
        	exit(EXIT_FAILURE);
    	}
*/
    	// Allocate the device output vector C
    	float *d_C = NULL;
    	errorFlag = cudaMalloc((void **)&d_C, sizeVector);

    	if (errorFlag != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(errorFlag));
        	exit(EXIT_FAILURE);
    	}

	    // Copy the host input vectors A and B in host memory to the device input vectors in device memory

	    errorFlag = cudaMemcpy(d_A, h_A, sizeVector, cudaMemcpyHostToDevice);

	    if (errorFlag != cudaSuccess)
	    {
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(errorFlag));
		exit(EXIT_FAILURE);
	    }


/*	    errorFlag = cudaMemcpy(d_B, h_B, sizeVector, cudaMemcpyHostToDevice);

	    if (errorFlag != cudaSuccess)
	    {
		fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(errorFlag));
		exit(EXIT_FAILURE);
	    }

*/
	    // Launch the Vector Add CUDA Kernel
	    int threadsPerBlock = n_threads_per_block;
	    int blocksPerGrid = n_threads/n_threads_per_block;

	    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
		
	    double t_time = 10000000;

	    for(int k=0; k<n_test; k++)
	    {
	    	startTime(&t_timer);
	    	bandwidthTest<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, numElements,n_elements_thread);
		cudaDeviceSynchronize();
	    	stopTime(&t_timer);
	    	errorFlag = cudaGetLastError();
	 
	    	if (errorFlag != cudaSuccess)
	    	{
			fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(errorFlag));
			exit(EXIT_FAILURE);
	    	}
		
		if(elapsedTime(t_timer) < t_time)
		{
			t_time = elapsedTime(t_timer);
		}

            }

	    // Copy the device result vector in device memory to the host result vector
	    // in host memory.
	    printf("Copy output data from the CUDA device to the host memory\n");
	    errorFlag = cudaMemcpy(h_C, d_C, sizeVector, cudaMemcpyDeviceToHost);

	    if (errorFlag != cudaSuccess)
	    {
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(errorFlag));
		exit(EXIT_FAILURE);
	    }


	    // Free device global memory
	    errorFlag = cudaFree(d_A);

/*
	    if (errorFlag != cudaSuccess)
	    {
		fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(errorFlag));
		exit(EXIT_FAILURE);
	    }
	    errorFlag = cudaFree(d_B);
*/
	    if (errorFlag != cudaSuccess)
	    {
		fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(errorFlag));
		exit(EXIT_FAILURE);
	    }
	    errorFlag = cudaFree(d_C);

	    if (errorFlag != cudaSuccess)
	    {
		fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(errorFlag));
		exit(EXIT_FAILURE);
	    }

	    // Free host memory
	    free(h_A);
//	    free(h_B);
	    free(h_C);

	    // Reset the device and exit
	    errorFlag = cudaDeviceReset();

	    if (errorFlag != cudaSuccess)
	    {
		fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(errorFlag));
		exit(EXIT_FAILURE);
	    }

	    double bw = 2.0*(1.0*n_elements/(1000.00*1000.00))/(t_time);

	    printf("Num Elements: %lu Time: %e Bandwidth(MB/s): %e\n",n_elements,t_time,bw);
	    return 0;
}
