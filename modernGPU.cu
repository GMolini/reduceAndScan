#include "modernGPU.cuh"
#include "stdio.h"

#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>


#define SHARED_SIZE 512
#define VT 3

void reduce_wrapper(uint numBlocks,
                    uint numThreads,
                    int* result,
                    int* vector,
                    int  vectorSize,
                    int vt){

    k_reduce <<<numBlocks,numThreads>>> (result, vector, vectorSize, vt);

}

__global__ void k_reduce(int* result, int* vector, int vectorSize, int vt){

    int gIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (gIdx >= vectorSize) return;

    int tIdx = threadIdx.x;
    __shared__ int s_vector[SHARED_SIZE];

    //Load values in shared memory
    int partial_sum = 0;
    for (int i= 0; i < VT; i++){

        int global_index = gIdx * VT + i;
        if (global_index < vectorSize)
            partial_sum += vector[global_index];

    }

    s_vector[tIdx] = partial_sum;
    __syncthreads();

    for (int i = blockDim.x; i >=1; i /= 2)
    {
        if (tIdx < i){
            s_vector[tIdx] += s_vector[i + tIdx];
        }
        __syncthreads();
    }

    //store value in global memory
    if (tIdx==0)
        result[blockIdx.x] = s_vector[tIdx];
}

__host__ __device__ uint iDivUp(uint a,
                                uint b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

__host__ __device__ void computeGridSize(uint n,
                                         uint blockSize,
                                         uint &numBlocks,
                                         uint &numThreads)
{
    uint min = blockSize;

    if (min > n)
        min = n;
    numThreads =min;
    numBlocks = iDivUp(n, numThreads);
}

void exclusiveScan_thrust(int *first,
                          int *last,
                          int *result,
                          int init)
{
    thrust::plus<int> binary_op;
    thrust::exclusive_scan(thrust::device,
                           first,
                           last,
                           result,
                           init,
                           binary_op);

}

void exclusiveScan_wrapper(uint numBlocks,
                           uint numThreads,
                           int* result,
                           int* vector,
                           int  vectorSize,
                           int vt){

    int* localScan;
    cudaMalloc((void **) &localScan, iDivUp(vectorSize,vt) * sizeof(int));

    int* interBlockScan;
    cudaMalloc((void **) &interBlockScan, numBlocks * sizeof(int));

    int* blockSums;
    cudaMalloc((void **) &blockSums, numBlocks * sizeof(int));

    //printf("numBlocks %d\n", numBlocks);

    //Perform scans in each CTA, and store the total number in each block in interBlockScan
    k_upsweep <<<numBlocks,numThreads>>> (localScan, blockSums, vector, vectorSize, vt, iDivUp(vectorSize,vt));

    uint numBlocks_exScan = iDivUp(numBlocks, numThreads);
    uint numThreads_exScan;
    //printf("Numblocks_exScan %d\n",numBlocks_exScan);
    computeGridSize(numBlocks,128,numBlocks_exScan,numThreads_exScan);

    if(numBlocks_exScan == 1)
        k_exclusiveScan <<< numBlocks_exScan, numThreads_exScan>>> (interBlockScan, blockSums, numBlocks, 1);
    else
        exclusiveScan_wrapper(numBlocks_exScan,
                              numThreads_exScan,
                              interBlockScan,
                              blockSums,
                              numBlocks,
                              VT);

    //Add to each block the carry-on of its respective block
    k_downsweep <<< numBlocks, numThreads >>> (result, vector, localScan, interBlockScan, vt, iDivUp(vectorSize,vt));

    cudaFree(localScan);
    cudaFree(interBlockScan);
    cudaFree(blockSums);
}

__global__ void k_downsweep(int* result,
                            int* originalArray,
                            int* parallelScans,
                            int* blocksExclusiveScan,
                            int vt,
                            int size)
{

    int gIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (gIdx >= size) return;

    int tIdx = threadIdx.x;
    int carryOn = 0;

    if (blockIdx.x != 0){
        carryOn = blocksExclusiveScan[blockIdx.x];
    }

    //Scan the VT values locally
    int localCarryOn = 0;
    if (tIdx != 0)
        localCarryOn = parallelScans[gIdx];

    int currentSum = 0;

    #pragma unroll
    for (int i=0; i < VT; i++)
    {
        if (tIdx == 0 && i==0)
            currentSum=0;
        else
            currentSum += originalArray[gIdx * VT + i - 1];
       result[gIdx*VT + i] = currentSum + localCarryOn + carryOn;
    }

}

__global__ void k_exclusiveScan(int* result, int*vector, int vectorSize, int vt)
{
    int gIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (gIdx >= vectorSize) return;

    int tIdx = threadIdx.x;
    __shared__ int s_vector[SHARED_SIZE];

    //Load values in shared memory
    int partial_sum = 0;

    #pragma unroll
    for (int i= 0; i < vt; i++){

        int global_index = gIdx * vt + i;
        if (global_index < vectorSize)
            partial_sum += vector[global_index];

    }

    s_vector[tIdx] = partial_sum;
    int first = 0;
    __syncthreads();

    #pragma unroll
    for (int offset = 1; offset < blockDim.x; offset += offset)
    {
        if (tIdx >= offset){
            partial_sum += s_vector[first + tIdx - offset];
        }
        first = blockDim.x - first;
        s_vector[first + tIdx] = partial_sum;
        __syncthreads();
    }

    if (tIdx != 0){
        result[gIdx] = s_vector[tIdx + first - 1];
    }else{
        result[gIdx] = 0;
    }
}



__global__ void k_upsweep(int* result, int* partialSums, int* vector, int vectorSize, int vt, int realSize){

    int gIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (gIdx >= realSize) return;

    int tIdx = threadIdx.x;
    __shared__ int s_vector[SHARED_SIZE];

    //Load values in shared memory
    int partial_sum = 0;

    #pragma unroll
    for (int i= 0; i < VT; i++){

        int global_index = gIdx * VT + i;
        if (global_index < vectorSize && (tIdx + i) != 0)
            partial_sum += vector[global_index - 1];
    }

    s_vector[tIdx] = partial_sum;
    int first = 0;
    __syncthreads();

    #pragma unroll
    for (int offset = 1; offset < blockDim.x; offset += offset)
    {
        if (tIdx >= offset){
            partial_sum += s_vector[first + tIdx - offset];
        }
        first = blockDim.x - first;
        s_vector[first + tIdx] = partial_sum;
        __syncthreads();
    }

    if (tIdx != 0){
        result[gIdx] = s_vector[tIdx + first - 1];
    }else{
        result[gIdx] = 0;
    }

    int lastElem = 0;
    if (blockIdx.x == 0)
        lastElem = vector[blockDim.x * VT-1];
    else if (blockIdx.x == gridDim.x -1)
        lastElem = vector[vectorSize-1];
    else
        lastElem = vector[((blockIdx.x +1) * blockDim.x * VT) -1];

    if (tIdx == 0)
        partialSums[blockIdx.x] = s_vector[blockDim.x + first - 1] + lastElem;

}

