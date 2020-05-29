#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace atspSolver
{
	__global__ void d_fillVectorRandomly(double *vector, int size);
	__global__ void d_copyVectorElements(double *dst, double *src, int size);
	__global__ void d_sumVectorElements(double *v1, double *v2, double *dst, int size);
	void displayMatrix(const double *matrix, int numberOfRows, int numberOfCols);
}
