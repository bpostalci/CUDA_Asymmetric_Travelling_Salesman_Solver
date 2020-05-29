#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#define CUDA_ERR_CHECK(code) { checkCudaErrors(code); }
const int blocksPerGrid = 128;
const int threadsPerBlock = 128;
void checkCudaErrors(cudaError err);