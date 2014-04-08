#ifndef MODERN_GPU_CUH
#define MODERN_GPU_CUH

#include <cuda_runtime.h>
#include <cuda.h>

void reduce_wrapper(uint numBlocks,
                    uint numThreads,
                    int * result,
                    int* vector,
                    int  vectorSize,
                    int vt);


__global__ void k_reduce(int * result, int* vector, int vectorSize, int vt);

void exclusiveScan_wrapper(uint numBlocks,
                           uint numThreads,
                           int* result,
                           int* vector,
                           int  vectorSize,
                           int vt);

void exclusiveScan_thrust(int *first,
                          int *last,
                          int *result,
                          int init);


__global__ void k_upsweep(int* result,
                          int* partialSums,
                          int* vector,
                          int vectorSize,
                          int vt,
                          int realSize);
__global__ void k_exclusiveScan(int* result, int*vector, int vectorSize, int vt);

__global__ void k_downsweep(int* result,
                            int* originalArray,
                            int* parallelScans,
                            int* blocksExclusiveScan,
                            int vt,
                            int size);

__host__ __device__ uint iDivUp(uint a,
                                uint b);
__host__ __device__ void computeGridSize(uint n,
                                         uint blockSize,
                                         uint &numBlocks,
                                         uint &numThreads);


#endif // PARTICLES_KERNEL_IMPL_CUH
