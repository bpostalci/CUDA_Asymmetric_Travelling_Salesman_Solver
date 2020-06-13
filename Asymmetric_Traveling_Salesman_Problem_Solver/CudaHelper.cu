#include "CudaHelper.cuh"

void checkCudaErrors(cudaError err)
{
	if (err > 0) {
		std::cerr << "Error: " << cudaGetErrorString(err);
		exit(err);
	}
}