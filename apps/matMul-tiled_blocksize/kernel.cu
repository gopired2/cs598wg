/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

#define TILE_SIZE 64

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/


/*  *** Basic version ***************
    int row = blockIdx.y * blockDim.y + threadIdx.y ;
    int col = blockIdx.x * blockDim.x + threadIdx.x ;
    if( (row < m) && (col < n)) 
    {
        float prod = 0 ;
        for (int i = 0; i < k; ++i)
        {
            prod += (A[row*k + i]) * (B[i*n + col]) ;
        }
        C[row*n + col] = prod ; 
    }
*/    

/********* TILED MEMORY Version ***********/
    //Allocate Shared Memory 

    __shared__ float shared_M[TILE_SIZE][TILE_SIZE] ;
    __shared__ float shared_N[TILE_SIZE][TILE_SIZE] ;

    //Note TILE_SIZE = BLOCK_SIZE = blockDim.y = blockDim.x 

    int row = blockIdx.y * blockDim.y + threadIdx.y ;
    int col = blockIdx.x * blockDim.x + threadIdx.x ;

    float prod = 0 ;

    for (int i = 0; i < (k+TILE_SIZE -1)/TILE_SIZE ; ++i )
    {
        //Check bounds and 
        //Load into Shared Memory 
        int curx = i*TILE_SIZE + threadIdx.x ;
        int cury = i*TILE_SIZE + threadIdx.y ;

        if ((row < m) && (curx  < k) )
            shared_M[threadIdx.y][threadIdx.x] = A[row*k + curx ] ;
        else 
            shared_M[threadIdx.y][threadIdx.x] = 0;

        if((col < n) && (cury < k))
            shared_N[threadIdx.y][threadIdx.x] = B[ cury*n + col] ;
        else
            shared_N[threadIdx.y][threadIdx.x] = 0 ;

        //Wait for all threads 
        __syncthreads() ;

        for(int j =0; j < TILE_SIZE ; ++j )
        {
            prod += shared_M[threadIdx.y][j] * shared_N[j][threadIdx.x] ;

        }
        //Wait for all the threads
        __syncthreads() ;
    }

    //Check for bounds and copy result

    if(row < m && col < n)
        C[row*n + col] = prod ;

}

void tiledSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'N') && (transb != 'n')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;


    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE) ;
    dim3 dimGrid((n + dimBlock.x -1)/dimBlock.x, (m + dimBlock.y - 1)/dimBlock.y ) ;


    // Invoke CUDA kernel -----------------------------------------------------

    
    mysgemm<<<dimGrid, dimBlock>>> (m, n, k, A, B, C) ;

    cudaThreadSynchronize();

}


