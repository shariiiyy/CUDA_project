#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cuda.h>
#define CUDA_SAFE_CALL_NO_SYNC( call) do {                              \
  cudaError err = call;                                                 \
  if( cudaSuccess != err) {                                             \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",       \
                __FILE__, __LINE__, cudaGetErrorString( err) );         \
    exit(EXIT_FAILURE);                                                 \
    } } while (0)

#define CUDA_SAFE_CALL( call) do {                                      \
  CUDA_SAFE_CALL_NO_SYNC(call);                                         \
  cudaError err = cudaThreadSynchronize();                              \
  if( cudaSuccess != err) {                                             \
     fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",      \
                 __FILE__, __LINE__, cudaGetErrorString( err) );        \
     exit(EXIT_FAILURE);                                                \
     } } while (0)


//input element size
const int N = 1024*512;
//block size
const int blocksize = 1024;


__global__ void maxBandwidth(int n, float* in, float* out){
  
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  
  if(i < n){
    float temp = in[i] + i * 2.0f;
    out[i] = out[i] + temp / (0.5f);
    
  }
}



int main(int argc, char **argv)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  

  //unsigned int num_threads = N;
  unsigned int numbytes = N * sizeof(float);
  
  //allocate host memory
  float *in = (float *) malloc(numbytes);
  float *out =(float *) malloc(numbytes);

  
  // initalize the memory
  for( unsigned int i = 0; i < N ; ++i)
    {
        in[i] = (float)i;
	out[i] = 3.0f;
    }

  //allocate device memory
  float *d_in, *d_out;
  CUDA_SAFE_CALL(cudaMalloc(&d_in, numbytes));
  CUDA_SAFE_CALL(cudaMalloc(&d_out, numbytes));
  CUDA_SAFE_CALL(cudaMemcpy(d_in, in, numbytes, cudaMemcpyHostToDevice));



  dim3  block(N/blocksize, 1, 1);
    //max block size(1024, 1024, 64)
  dim3  thread(blocksize, 1 ,1);

  // execute the kernel
  cudaEventRecord(start, 0);
  maxBandwidth<<< block, thread, numbytes>>>(N, d_in, d_out);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  // check if kernel execution generated and error
  // CUT_CHECK_ERROR("Kernel execution failed");


  CUDA_SAFE_CALL( cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost));


  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("\nProcessing time: %f (ms)\n", elapsedTime);
  printf("Effective Bandwidth (GB/s): %f\n", (numbytes*3)/elapsedTime/1e6);
  // printf("Total number of memory read/write on GPU (bytes): %d\n\n", numbytes);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // cleanup memory
  free(in);
  free(out);
  CUDA_SAFE_CALL(cudaFree(d_in));
  CUDA_SAFE_CALL(cudaFree(d_out));

}
