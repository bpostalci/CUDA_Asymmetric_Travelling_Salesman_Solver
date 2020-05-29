#include "VectorHelper.cuh"
#include <curand.h>
#include <curand_kernel.h>
#include <sstream>
#include <iostream>

namespace atspSolver
{
	// device functions - functions that can be used for another purposes
	__global__ void d_fillVectorRandomly(double *vector, int size)
	{
		curandState_t state;

		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		curand_init(tid,           /* the seed controls the sequence of random values that are produced */
			threadIdx.x,		   /* the sequence number is only important with multiple cores */
			0,					   /* the offset is how much extra we advance in the sequence for each call, can be 0 */
			&state);


		while (tid < size)
		{
			vector[tid] = curand(&state) % 1000 + 50;
			tid += blockDim.x * gridDim.x;
		}
	}

	__global__ void d_copyVectorElements(double *dst, double *src, int size)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		while (tid < size)
		{
			dst[tid] = src[tid];
			tid += blockDim.x * gridDim.x;
		}
	}

	__global__ void d_sumVectorElements(double *v1, double *v2, double *dst, int size)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		while (tid < size)
		{
			dst[tid] = v1[tid] + v2[tid];
			tid += blockDim.x * gridDim.x;
		}
	}
	// device functions

	// host functions
	void displayMatrix(const double *matrix, int numberOfRows, int numberOfCols)
	{
		std::stringstream stream;
		for (int i = 0; i < numberOfRows; i++)
		{
			for (int j = 0; j < numberOfCols; j++)
			{
				int index = j + i * numberOfCols;
				stream << "m[" << index << "]: " << matrix[index] << ", ";
			}
			stream << std::endl;
		}
		stream.seekp(-3, std::ios_base::end); stream << ' ';
		std::cout << stream.str();
	}
	// host functions

}