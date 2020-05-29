#include "CudaHelper.cuh"

void checkCudaErrors(cudaError err)
{
	if (err > 0) {
		std::cout << "Error: " << cudaGetErrorString(err);
		exit(err);
	}
}