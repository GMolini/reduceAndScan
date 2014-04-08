#include <QCoreApplication>

#include <stdio.h>
#include <iostream>

#include "modernGPU.cuh"

#define VT 3

void pickCudaDevice()
{
    //Get Cuda architecture as defined in compilation .pro file.
    QString cudaArch(CUDA_ARCH);

    //Get all available devices
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    int device, selectedDevice = -1;

    //Loop through cuda devices
    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        QString deviceArchitecture = QString("sm_%1%2").arg(deviceProp.major).arg(deviceProp.minor);

        //Select device that has the desired compilation architecture
        if (cudaArch == deviceArchitecture)
            selectedDevice = device;
    }

    if (selectedDevice != -1){
        printf(" Set cuda Device \n");
        cudaSetDevice(selectedDevice);
    }else{
        printf("pickCudaDevice: Could not find a device with the compilation set cuda capabilities!\n");
        exit(0);
    }

    cudaDeviceReset();

}


void checkCudaError(){
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError  != cudaSuccess){
        printf("Cuda error msg: %s\n",cudaGetErrorString(cudaError));
        exit(1);
    }
}

int areVectorsEqual(int* v1, int* v2, int size){
    for(int i=0; i<size; i++){
        if(v1[i]!=v2[i])
            return 1;
    }
    return 0;
}

int vectorsDifference(int* v1, int* v2, int size){
    for(int i=0; i<size; i++){
        if(v1[i] - v2[i] !=0)
            return i;
    }
    return 0;
}



int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);


    int size = 100000000;
    int vt = 3;

    int* vector = (int *) malloc (size * sizeof(int));
    int* vectorCheck = (int *) malloc (size * sizeof(int));

    int number = 0;
    for (int i = 0; i < size; i++){
        if (i % (vt * 128) == 0)
            number++;
        vector[i] = number; //rand() % 10;
    }

/*
    for (int i=0; i<size; i++)
        printf(" %d ", vector[i]);
   printf("\n");
*/

    pickCudaDevice();
    checkCudaError();

    int* d_vector;
    cudaMalloc((void **) &d_vector, size * sizeof(int));
    checkCudaError();

    int* d_result;
    cudaMalloc((void **) &d_result, size * sizeof(int));
    checkCudaError();

    int* d_vectorCheck;
    cudaMalloc((void **) &d_vectorCheck, size * sizeof(int));
    checkCudaError();

    int* d_resultCheck;
    cudaMalloc((void **) &d_resultCheck, size * sizeof(int));
    checkCudaError();

    uint numThreads, numBlocks;

    cudaMemcpy(d_vector,vector,size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vectorCheck,vector,size * sizeof(int), cudaMemcpyHostToDevice);


    computeGridSize(iDivUp(size,VT),128,numBlocks,numThreads);

    printf("Start kernel\n");
    //reduce_wrapper(numBlocks,numThreads,d_result,d_vector,size, vt);
    //checkCudaError();

    //gpu time measurement
    cudaEvent_t gstart_exScan,gstop_exScan;
    cudaEventCreate(&gstart_exScan);
    cudaEventCreate(&gstop_exScan);

    cudaEventRecord(gstart_exScan, 0);

    exclusiveScan_wrapper(numBlocks,
                          numThreads,
                          d_result,
                          d_vector,
                          size,
                          VT);

    cudaEventRecord(gstop_exScan, 0);
    cudaEventSynchronize(gstop_exScan);

    float gpu_time_exScan;
    cudaEventElapsedTime(&gpu_time_exScan, gstart_exScan, gstop_exScan);
    printf("Our GPU version has finished, it took %f ms\n",gpu_time_exScan );

    cudaEventDestroy(gstart_exScan); //cleaning up a bit
    cudaEventDestroy(gstop_exScan);
    checkCudaError();


    //gpu time measurement
    cudaEvent_t gstart,gstop;
    cudaEventCreate(&gstart);
    cudaEventCreate(&gstop);

    cudaEventRecord(gstart, 0);

    exclusiveScan_thrust(d_vectorCheck,
                         d_vectorCheck + size,
                         d_resultCheck,
                         0);

    cudaEventRecord(gstop, 0);
    cudaEventSynchronize(gstop);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, gstart, gstop);
    printf("Thrust version has finished, it took %f ms\n",gpu_time );

    cudaEventDestroy(gstart); //cleaning up a bit
    cudaEventDestroy(gstop);

    checkCudaError();

    printf("End kernel\n");

    cudaMemcpy(vector,d_result,size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(vectorCheck,d_resultCheck,size * sizeof(int), cudaMemcpyDeviceToHost);

    /*
    for (int i=0; i<size; i++)
        printf(" %d ", vectorCheck[i]);
   printf("\n");
*/

/*
   for (int i=0; i<size; i++)
       printf(" %d ", vector[i]);
  printf("\n");
*/
    printf("Difference %d\n", vectorsDifference(vector,vectorCheck,size));

    if(areVectorsEqual(vector,vectorCheck,size) == 0)
        printf("Vectors are equal!!\n");
    else
        printf("Vectors are NOT equal :( \n");
    checkCudaError();

    cudaFree(d_vector);
    free(vector);
    cudaFree(d_result);
    cudaFree(d_resultCheck);
    cudaFree(d_vectorCheck);
    free(vectorCheck);
    return 0;
}
