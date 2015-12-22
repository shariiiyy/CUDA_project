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
const int N = 1024*1024*32;
//block size
const int blocksize = 1024;


__global__ void maxBandwidth(int n, float* in, float* out){
  
  int i = blockIdx.x*blockDim.x + threadIdx.x;
 
  if(i < n){
    in[i] = in[i] + 4.0f;   //5.0
    out[i] = out[i] + in[i];//5.0
    in[i] = in[i] - 4.0f;   //1.0
    out[i] = out[i] - in[i];//4.0
    in[i] = in[i] + 1.0f;   //2.0
    out[i] = out[i] + in[i];//6.0
  }
  /*if(threadIdx.x == 0 && blockIdx.x == 0){
	printf("%d\n", threadIdx.x);
  }
  */
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
        in[i] = 1.0f;
	out[i] = 0.0f;
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
  maxBandwidth<<< block, thread>>>(N, d_in, d_out);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  // copy output to host memory
  CUDA_SAFE_CALL( cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost));
  
  //check output from kernel
  int flag = 1;
  for(unsigned int j=0; j<N; j++){
 	if(out[j] != 6.0 ){
		printf("out[%d]: %f\n", j, out[j]);
		flag = 0;
	}
  }
  if(flag == 1){
	printf("ALL SUCCESS!\n");
  }else{
	printf("WRONG!!!\n");
  }
  
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("\nProcessing time: %f (ms)\n", elapsedTime);
  printf("Effective Bandwidth (GB/s): %f\n\n", (12*numbytes)/elapsedTime/1e6);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // cleanup memory
  free(in);
  free(out);
  CUDA_SAFE_CALL(cudaFree(d_in));
  CUDA_SAFE_CALL(cudaFree(d_out));

}
