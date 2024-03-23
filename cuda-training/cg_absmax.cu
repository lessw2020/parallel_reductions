// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cstdlib>
#include <ctime>

// includes, project
#include <helper_functions.h>
#include <helper_cuda.h>

#include <cuda_runtime.h>

const char *sSDKsample = "reductionMultiBlockCG";

#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <algorithm>
#include <limits>

namespace cg = cooperative_groups;

/*
  Parallel sum reduction using shared memory
  - takes log(n) steps for n input elements
  - uses n/2 threads
  - only works for power-of-2 arrays

  This version adds multiple elements per thread sequentially. This reduces the
  overall cost of the algorithm while keeping the work complexity O(n) and the
  step complexity O(log n).
  (Brent's Theorem optimization)

  See the CUDA SDK "reduction" sample for more information.
*/
using namespace cooperative_groups;

__device__ void reduceBlock(float* sdata, const thread_block &cgb) {
    const unsigned int tid = cgb.thread_rank();
    thread_block_tile<32> tile32 = tiled_partition<32>(cgb);

    //effectively a warp shuffle down synch
    sdata[tid] = reduce(tile32, sdata[tid], greater<float>());
    cgb.sync();
    float alpha = 0;
    //collect all 0 elems from each tile reduction
    if (cgb.thread_rank() ==0) {
        for (int i=0; i< blockDim.x; i+= tile32.size()) {
            alpha = max(abs(alpha), abs(sdata[i]));
        }
        sdata[0] = alpha;
    }
    cgb.sync();


}
/*
__device__ void reduceBlock(float* sdata, const thread_block& cgb) {
    const unsigned int tid = cgb.thread_rank();
    thread_block_tile<32> tile32 = tiled_partition<32>(cgb);

    // Perform reduction within each warp using shuffle instructions
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
        float other = tile32.shfl_down(sdata[tid], offset);
        sdata[tid] = max(sdata[tid], other);
    }

    // Collect the maximum value from each warp
    if (tile32.thread_rank() == 0) {
        sdata[tid / tile32.size()] = sdata[tid];
    }
    cgb.sync();

    // Perform final reduction across warps
    if (tid < blockDim.x / tile32.size()) {
        for (int offset = (blockDim.x / tile32.size()) / 2; offset > 0; offset /= 2) {
            float other = (tid + offset < blockDim.x / tile32.size()) ? sdata[tid + offset] : -INFINITY;
            sdata[tid] = max(sdata[tid], other);
        }
    }
    cgb.sync();
}
*/
/*
__device__ void reduceBlock(float *sdata, const cg::thread_block &cta) {
  const unsigned int tid = cta.thread_rank();
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  sdata[tid] = cg::reduce(tile32, sdata[tid], cg::greater<float>());
  cg::sync(cta);

  float beta = 0.0;
  if (cta.thread_rank() == 0) {
    beta = -INFINITY;
    for (int i = 0; i < blockDim.x; i += tile32.size()) {
      //beta += sdata[i];
      beta = max(beta, sdata[i]);
    }
    sdata[0] = beta;
  }
  cg::sync(cta);
}
*/

// This reduction kernel reduces an arbitrary size array in a single kernel
// invocation
//
// For more details on the reduction algorithm (notably the multi-pass
// approach), see the "reduction" sample in the CUDA SDK.
__global__ void reduceSinglePassMultiBlockCG(const float *g_idata,
                                                        float *g_odata,
                                                        unsigned int n) {
  // Handle to thread block group
  thread_block block = this_thread_block();
  grid_group grid = this_grid();

  extern float __shared__ sdata[];

  sdata[block.thread_rank()] = 0; //std::numeric_limits<float>::lowest();
  for (int i = grid.thread_rank(); i < n; i += grid.size()) {
    sdata[block.thread_rank()] = max(abs(sdata[block.thread_rank()]), abs(g_idata[i]));
}

  block.sync();

  // Reduce each block (called once per block)
  reduceBlock(sdata, block);
  // Write out the result to global memory
  if (block.thread_rank() == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }
  grid.sync();

  if (grid.thread_rank() == 0) {
    for (int block = 1; block < gridDim.x; block++) {
      g_odata[0] = max(abs(g_odata[0]), abs(g_odata[block]));
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
void call_reduceSinglePassMultiBlockCG(int size, int threads, int numBlocks,
                                       float *d_idata, float *d_odata) {
  int smemSize = threads * sizeof(float);
  void *kernelArgs[] = {
      (void *)&d_idata, (void *)&d_odata, (void *)&size,
  };

  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(numBlocks, 1, 1);

  cudaLaunchCooperativeKernel((void *)reduceSinglePassMultiBlockCG, dimGrid,
                              dimBlock, kernelArgs, smemSize, NULL);
  // check if kernel execution generated an error
  getLastCudaError("Kernel execution failed");
}

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runTest(int argc, char **argv, int device);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  cudaDeviceProp deviceProp = {0};
  int dev;

  printf("%s Starting...\n\n", sSDKsample);

  dev = findCudaDevice(argc, (const char **)argv);
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
  if (!deviceProp.cooperativeLaunch) {
    printf(
        "\nSelected GPU (%d) does not support Cooperative Kernel Launch, "
        "Waiving the run\n",
        dev);
    exit(EXIT_WAIVED);
  }

  bool bTestPassed = false;
  bTestPassed = runTest(argc, argv, dev);


  if (bTestPassed) {
      std::cout << "Test passed!" << std::endl;
  } else {
      std::cout << "Test failed!" << std::endl;
  }

  exit(bTestPassed ? EXIT_SUCCESS : EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////
//! Compute sum reduction on CPU
//! We use Kahan summation for an accurate sum of large arrays.
//! http://en.wikipedia.org/wiki/Kahan_summation_algorithm
//!
//! @param data       pointer to input data
//! @param size       number of input data elements
////////////////////////////////////////////////////////////////////////////////
/*template <class T>
T reduceCPU(T *data, int size) {
  T sum = data[0];
  T c = (T)0.0;

  for (int i = 1; i < size; i++) {
    T y = data[i] - c;
    T t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }

  return sum;
}
*/
template <class T>
T reduceCPU(T* data, int size) {
    T max_value = data[0];
    for (int i = 1; i < size; i++) {
        max_value = std::max(max_value, data[i]);
    }
    return max_value;
}

unsigned int nextPow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the reduction
// We set threads / block to the minimum of maxThreads and n/2.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks,
                            int &threads) {
  if (n == 1) {
    threads = 1;
    blocks = 1;
  } else {
    checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(
        &blocks, &threads, reduceSinglePassMultiBlockCG));
  }

  blocks = min(maxBlocks, blocks);
}

////////////////////////////////////////////////////////////////////////////////
// This function performs a reduction of the input data multiple times and
// measures the average reduction time.
////////////////////////////////////////////////////////////////////////////////
float benchmarkReduce(int n, int numThreads, int numBlocks, int maxThreads,
                      int maxBlocks, int testIterations,
                      StopWatchInterface *timer, float *h_odata, float *d_idata,
                      float *d_odata) {
  float gpu_result = 0;
  cudaError_t error;

  printf("\nLaunching %s kernel\n",
         "SinglePass Multi Block Cooperative Groups");
  for (int i = 0; i < testIterations; ++i) {
    gpu_result = 0;
    sdkStartTimer(&timer);
    call_reduceSinglePassMultiBlockCG(n, numThreads, numBlocks, d_idata,
                                      d_odata);
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
  }

  // copy final sum from device to host
  error =
      cudaMemcpy(&gpu_result, d_odata, sizeof(float), cudaMemcpyDeviceToHost);
  checkCudaErrors(error);

  return gpu_result;
}

////////////////////////////////////////////////////////////////////////////////
// The main function which runs the reduction test.
////////////////////////////////////////////////////////////////////////////////
bool runTest(int argc, char **argv, int device) {
  int size = 1 << 25;  // number of elements to reduce
  bool bTestPassed = false;

  if (checkCmdLineFlag(argc, (const char **)argv, "n")) {
    size = getCmdLineArgumentInt(argc, (const char **)argv, "n");
  }

  printf("%d elements\n", size);

  // Set the device to be used
  cudaDeviceProp prop = {0};
  checkCudaErrors(cudaSetDevice(device));
  checkCudaErrors(cudaGetDeviceProperties(&prop, device));

  // create random input data on CPU
  unsigned int bytes = size * sizeof(float);

  float *h_idata = (float *)malloc(bytes);


    srand(static_cast<unsigned int>(time(0)));

    // Generate random numbers
    for (int i = 0; i < size; i++) {
        h_idata[i] = rand() % 201 - 100;
    }

    h_idata[111] = -1010;


  //for (int i = 0; i < size; i++) {
    // Keep the numbers small so we don't get truncation error in the sum
  //  h_idata[i] = curand_uniform(&state) * 2.0f - 1.0f; // (rand()); // & 0xFF) / (float)RAND_MAX;
  //}

  // Determine the launch configuration (threads, blocks)
  int maxThreads = 0;
  int maxBlocks = 0;

  if (checkCmdLineFlag(argc, (const char **)argv, "threads")) {
    maxThreads = getCmdLineArgumentInt(argc, (const char **)argv, "threads");
  } else {
    maxThreads = prop.maxThreadsPerBlock;
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "maxblocks")) {
    maxBlocks = getCmdLineArgumentInt(argc, (const char **)argv, "maxblocks");
  } else {
    maxBlocks = prop.multiProcessorCount *
                (prop.maxThreadsPerMultiProcessor / prop.maxThreadsPerBlock);
  }

  int numBlocks = 0;
  int numThreads = 0;
  getNumBlocksAndThreads(size, maxBlocks, maxThreads, numBlocks, numThreads);

  // We calculate the occupancy to know how many block can actually fit on the
  // GPU
  int numBlocksPerSm = 0;
  checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm, reduceSinglePassMultiBlockCG, numThreads,
      numThreads * sizeof(float)));

  int numSms = prop.multiProcessorCount;
  if (numBlocks > numBlocksPerSm * numSms) {
    numBlocks = numBlocksPerSm * numSms;
  }
  printf("numThreads: %d\n", numThreads);
  printf("numBlocks: %d\n", numBlocks);

  // allocate mem for the result on host side
  float *h_odata = (float *)malloc(numBlocks * sizeof(float));

  // allocate device memory and data
  float *d_idata = NULL;
  float *d_odata = NULL;

  checkCudaErrors(cudaMalloc((void **)&d_idata, bytes));
  checkCudaErrors(cudaMalloc((void **)&d_odata, numBlocks * sizeof(float)));

  // copy data directly to device memory
  checkCudaErrors(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_odata, h_idata, numBlocks * sizeof(float),
                             cudaMemcpyHostToDevice));

  int testIterations = 100;

  StopWatchInterface *timer = 0;
  sdkCreateTimer(&timer);

  float gpu_result = 0;

  gpu_result =
      benchmarkReduce(size, numThreads, numBlocks, maxThreads, maxBlocks,
                      testIterations, timer, h_odata, d_idata, d_odata);

  float reduceTime = sdkGetAverageTimerValue(&timer);
  printf("Average time: %f ms\n", reduceTime);
  printf("Bandwidth:    %f GB/s\n\n",
         (size * sizeof(int)) / (reduceTime * 1.0e6));

  // compute reference solution
  float cpu_result = reduceCPU<float>(h_idata, size);
  printf("GPU result = %0.12f\n", gpu_result);
  printf("CPU result = %0.12f\n", cpu_result);

  float threshold = 1e-8 * size;
  float diff = abs((float)gpu_result - (float)cpu_result);
  bTestPassed = (diff < threshold);

  // cleanup
  sdkDeleteTimer(&timer);

  free(h_idata);
  free(h_odata);
  cudaFree(d_idata);
  cudaFree(d_odata);

  return bTestPassed;
}
