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
#define ITERATIONS 1024

using namespace std;


__global__ void cache_latency(unsigned int *latency, int *data)
{
        //long unsigned tid = blockDim.x * blockIdx.x + (threadIdx.x);
	//int i;

	//__shared__ int dump[ITERATIONS];

	//__shared__ int s_starttime[ITERATIONS];
	//__shared__ int s_stoptime[ITERATIONS];	

	unsigned int start_t, stop_t;
	int local;	

	//__syncthreads();
		
	//for(int i=0; i<ITERATIONS; i++)
        //{
	//	dump[i] = data[i];
		//__syncthreads();
	//}

	local = 0;	
	for(int i=0; i<ITERATIONS; i++)
	{
		start_t = clock();
		
		//#pragma unroll	
		for(int j=0; j<1024; j++)
			local += data[j];
		stop_t = clock();
		//__syncthreads();	
		
		data[i] = local+1;

		//s_starttime[i] = start_t;
		//s_stoptime[i] = stop_t;		
		//__syncthreads();

		latency[i] = start_t;
                latency[i + ITERATIONS] = stop_t;

	}	

/*
	for(int i=0; i<ITERATIONS; i++)
       	{
		latency[i] = s_starttime[i];// start_t;
		latency[i + ITERATIONS] = s_stoptime[i];// stop_t;
	}
*/
	
}

int main(int argc, char **argv)
{

	int data[8192];
	
	srand(12);	

	for(int i=0; i<8192; i++)
	{
		data[i] = i;//rand();
	}
	
	unsigned int latency[2*ITERATIONS];
	unsigned int *d_latency;
	int *d_data;
	cudaError_t errorFlag = cudaSuccess;
	
	errorFlag = cudaMalloc((void**) &d_latency, (sizeof(unsigned int)*2*ITERATIONS));	

	if(errorFlag != cudaSuccess)                                                                  
        {       
                fprintf(stderr, "Failed to alloc memory (error code %s)!\n", cudaGetErrorString(errorFlag));                                                                                      
                exit(-1);                                                                             
        } 

	errorFlag = cudaMalloc((void**) &d_data, (sizeof(int)*8192));

        if(errorFlag != cudaSuccess)
        {
                fprintf(stderr, "Failed to alloc memory (error code %s)!\n", cudaGetErrorString(errorFlag));
                exit(-1);
        }

	errorFlag = cudaMemcpy(d_data, data,  (sizeof(int)*8192), cudaMemcpyHostToDevice);

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

	errorFlag = cudaMemcpy(latency, d_latency,  (sizeof(int)*2*ITERATIONS), cudaMemcpyDeviceToHost);

	if(errorFlag != cudaSuccess)                                                                  
        {       
                fprintf(stderr, "Failed to copyback (error code %s)!\n", cudaGetErrorString(errorFlag));                                                                                      
                exit(-1);                                                                             

        } 	

	cout<<"Start times\n";
	for(int i=0; i<ITERATIONS; i++)
	{
		cout<<latency[i]<<" ";
	}

	cout<<"\nStop times\n";
        for(int i=0; i<ITERATIONS; i++)
        {
                cout<<latency[i+ITERATIONS]<<" ";
        }
	
	cout<<"\nLatency\n";
        for(int i=0; i<ITERATIONS; i++)
        {
                cout<<latency[i+ITERATIONS] - latency[i]<<" ";
        }
	
	cout<<endl;

	
	return 0;
}
