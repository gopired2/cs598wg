/***
*	Ashutosh Dhar
*	Department of Electrical and Computer Engineeing
*	University of Illinois, Urbana-Champaign
*	
*/

#include <cuda.h>
#include <iostream>
#include <cstdio>

#define THREADS_PER_SM 1
#define BLOCKS_PER_SM 1
#define ITERATIONS 64
#define L2_CACHE_SIZE 512*1024
#define DATA_SIZE (L2_CACHE_SIZE * ITERATIONS)

using namespace std;


__global__ void cache_latency(unsigned int *latency, float *data)
{

	unsigned int start_t, stop_t;
	float local;	
	int load=0;	

	for(int i=0; i<DATA_SIZE; i++)
	{
		start_t = clock();
		local = data[load];
		stop_t = clock();
		__syncthreads();

		data[load] = local+1;

		latency[i] = start_t;
		latency[i + ITERATIONS] = stop_t;
	}
	
}

int main(int argc, char **argv)
{

	float *data;
	data = (float*) malloc(sizeof(float)*DATA_SIZE);
	
	srand(12);	

	for(int i=0; i<DATA_SIZE; i++)
	{
		data[i] = 1.0*rand();
	}
	
	unsigned int *latency;

	latency = (unsigned int*) malloc((sizeof(int)) * 2 * DATA_SIZE);
	
	unsigned int *d_latency;
	float *d_data;
	cudaError_t errorFlag = cudaSuccess;
	
	errorFlag = cudaMalloc((void**) &d_latency, (sizeof(unsigned int)*2*DATA_SIZE));	

	if(errorFlag != cudaSuccess)                                                                  
        {       
                fprintf(stderr, "Failed to alloc memory (error code %s)!\n", cudaGetErrorString(errorFlag));                                                                                      
                exit(-1);                                                                             
        } 

	errorFlag = cudaMalloc((void**) &d_data, (sizeof(float)*DATA_SIZE));

        if(errorFlag != cudaSuccess)
        {
                fprintf(stderr, "Failed to alloc memory (error code %s)!\n", cudaGetErrorString(errorFlag));
                exit(-1);
        }

	errorFlag = cudaMemcpy(d_data, data,  (sizeof(float)*DATA_SIZE), cudaMemcpyHostToDevice);

        if(errorFlag != cudaSuccess)
        {
                fprintf(stderr, "Failed to copyback (error code %s)!\n", cudaGetErrorString(errorFlag));
                exit(-1);

        }


	dim3	dimBlock(THREADS_PER_SM,1,1);
	dim3 	dimGrid(BLOCKS_PER_SM,1,1);

	cache_latency<<<dimGrid,dimBlock>>>(d_latency,d_data);
	cudaDeviceSynchronize();

	errorFlag = cudaGetLastError();
	
	if(errorFlag != cudaSuccess) 	
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(errorFlag));
		exit(-1);
	}	

	errorFlag = cudaMemcpy(latency, d_latency,  (sizeof(int)*2*DATA_SIZE), cudaMemcpyDeviceToHost);

	if(errorFlag != cudaSuccess)                                                                  
        {       
                fprintf(stderr, "Failed to copyback (error code %s)!\n", cudaGetErrorString(errorFlag));                                                                                      
                exit(-1);                                                                             

        } 	

	
	cout<<"\nLatency\n";
        for(int i=0; i<DATA_SIZE; i++)
        {
                cout<<latency[i+ITERATIONS] - latency[i]<<" ";
        }
	
	cout<<endl;

	
	return 0;
}
