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
int ITERATIONS;
int L2_CACHE_SIZE = 512*1024;
int DATA_SIZE;// (L2_CACHE_SIZE * ITERATIONS)

using namespace std;


__global__ void cache_latency(double *latency, int *data, int DATA_SIZE)
{
	
	//__shared__ double sh_start;
	//__shared__ double sh_stop;
	__shared__ long long int run_latency;
	
	unsigned int start_t, stop_t;
	//float local;	
	int load=0;	

	for(int i=0; i<DATA_SIZE; i++)
	{
		start_t = clock();
		load = data[load];
		stop_t = clock();
		__syncthreads();

		//data[load] = local + 1;
		
		run_latency += (stop_t - start_t);	
		__syncthreads();
	}


	latency[0] = (double)(run_latency)/(DATA_SIZE);
	
	
}

int main(int argc, char **argv)
{
	if(argc <2)
	{
		cerr<<"Enter iterations!";
		return -1;
	}
	
	ITERATIONS = atoi(argv[1]);
	DATA_SIZE = L2_CACHE_SIZE * ITERATIONS;

	//double sum;
	int *data;
	data = (int*) malloc(sizeof(int)*DATA_SIZE);
	
	srand(12);	

	for(int i=0; i<DATA_SIZE; i++)
	{
		data[i] = i;//1.0*rand();
	}
	
	double *latency;

	latency = (double*) malloc((sizeof(double)) *1);
	
	double *d_latency;
	int *d_data;
	cudaError_t errorFlag = cudaSuccess;
	
	errorFlag = cudaMalloc((void**) &d_latency, (sizeof(double)*1));	

	if(errorFlag != cudaSuccess)                                                                  
        {       
                fprintf(stderr, "Failed to alloc memory (error code %s)!\n", cudaGetErrorString(errorFlag));                                                                                      
                exit(-1);                                                                             
        } 

	errorFlag = cudaMalloc((void**) &d_data, (sizeof(int)*DATA_SIZE));

        if(errorFlag != cudaSuccess)
        {
                fprintf(stderr, "Failed to alloc memory (error code %s)!\n", cudaGetErrorString(errorFlag));
                exit(-1);
        }

	errorFlag = cudaMemcpy(d_data, data,  (sizeof(int)*DATA_SIZE), cudaMemcpyHostToDevice);

        if(errorFlag != cudaSuccess)
        {
                fprintf(stderr, "Failed to copyback (error code %s)!\n", cudaGetErrorString(errorFlag));
                exit(-1);

        }


	dim3	dimBlock(THREADS_PER_SM,1,1);
	dim3 	dimGrid(BLOCKS_PER_SM,1,1);

	cache_latency<<<dimGrid,dimBlock>>>(d_latency,d_data,DATA_SIZE);
	cudaDeviceSynchronize();

	errorFlag = cudaGetLastError();
	
	if(errorFlag != cudaSuccess) 	
	{
		fprintf(stderr, "Kernel launch error!  (error code %s)!\n", cudaGetErrorString(errorFlag));
		exit(-1);
	}	

	errorFlag = cudaMemcpy(latency, d_latency,  (sizeof(double)*1), cudaMemcpyDeviceToHost);

	if(errorFlag != cudaSuccess)                                                                  
        {       
                fprintf(stderr, "Failed to copyback (error code %s)!\n", cudaGetErrorString(errorFlag));                                                                                      
                exit(-1);                                                                             

        } 	

	
	cout<<"\nLatency\n";
        //for(int i=0; i<1; i++)
        //{
        //       sum+=latency[i+ITERATIONS] - latency[i];
        //}

	cout<<": "<< latency[0]<<endl;	

	cout<<endl;

	
	return 0;
}
