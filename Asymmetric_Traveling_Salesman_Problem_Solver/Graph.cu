#include "Graph.cuh"
#include "CudaHelper.cuh"
#include "VectorHelper.cuh"

namespace atspSolver
{
	Graph::Graph(int numberOfNodes) : numberOfNodes_(numberOfNodes), adjacencyMatrix_(new double[numberOfNodes*numberOfNodes])
	{
		Graph::generateGraph();
	}

	Graph::Graph(const double *adjacencyMatrix, int numberOfNodes) : numberOfNodes_(numberOfNodes), adjacencyMatrix_(new double[numberOfNodes*numberOfNodes])
	{
		memcpy(this->adjacencyMatrix_, adjacencyMatrix, sizeof(double) * numberOfNodes * numberOfNodes);
	}

	Graph::~Graph()
	{
		delete[] adjacencyMatrix_;
	}

	void Graph::copyElementsFromGraph(const Graph& graph)
	{
		double *devAdjacencyMatrixSrc, *devAdjacencyMatrixDst;
		unsigned int totalBytes = sizeof(double) * numberOfNodes_ * numberOfNodes_;

		CUDA_ERR_CHECK(cudaMalloc((void**)&devAdjacencyMatrixSrc, totalBytes));
		CUDA_ERR_CHECK(cudaMalloc((void**)&devAdjacencyMatrixDst, totalBytes));
		CUDA_ERR_CHECK(cudaMemcpy((void *)devAdjacencyMatrixSrc, (const void *)graph.adjacencyMatrix_, totalBytes, cudaMemcpyHostToDevice));
		d_copyVectorElements << <blocksPerGrid, threadsPerBlock >> > (devAdjacencyMatrixDst, devAdjacencyMatrixSrc, numberOfNodes_ * numberOfNodes_);
		CUDA_ERR_CHECK(cudaMemcpy(adjacencyMatrix_, devAdjacencyMatrixDst, totalBytes, cudaMemcpyDeviceToHost));

		CUDA_ERR_CHECK(cudaFree(devAdjacencyMatrixSrc));
		CUDA_ERR_CHECK(cudaFree(devAdjacencyMatrixDst));
	}

	Graph::Graph(const Graph& graph)
	{
		adjacencyMatrix_ = new double[graph.numberOfNodes_ * graph.numberOfNodes_];
		numberOfNodes_ = graph.numberOfNodes_;

		copyElementsFromGraph(graph);
	}

	Graph& Graph::operator=(const Graph& graph)
	{
		double *origAdjacencyMatrix = adjacencyMatrix_;
		adjacencyMatrix_ = new double[graph.numberOfNodes_ * graph.numberOfNodes_];
		numberOfNodes_ = graph.numberOfNodes_;

		copyElementsFromGraph(graph);

		delete[] origAdjacencyMatrix;

		return *this;
	}

	double Graph::operator()(int row, int col) const
	{
		int index = col + row * numberOfNodes_;
		return this->adjacencyMatrix_[index];
	}

	double Graph::operator[](int index) const
	{
		return this->adjacencyMatrix_[index];
	}

	Graph Graph::operator+(const Graph& graph) const
	{
		if (numberOfNodes_ != graph.numberOfNodes_)
		{
			throw;
		}
		double *devAdjacencyMatrix1, *devAdjacencyMatrix2, *devAdjacencyMatrixDst;
		unsigned int totalBytes = sizeof(double) * numberOfNodes_ * numberOfNodes_;

		CUDA_ERR_CHECK(cudaMalloc((void**)&devAdjacencyMatrix1, totalBytes));
		CUDA_ERR_CHECK(cudaMalloc((void**)&devAdjacencyMatrix2, totalBytes));
		CUDA_ERR_CHECK(cudaMalloc((void**)&devAdjacencyMatrixDst, totalBytes));
		CUDA_ERR_CHECK(cudaMemcpy((void *)devAdjacencyMatrix1, (const void *)graph.adjacencyMatrix_, totalBytes, cudaMemcpyHostToDevice));
		CUDA_ERR_CHECK(cudaMemcpy((void *)devAdjacencyMatrix2, (const void *)adjacencyMatrix_, totalBytes, cudaMemcpyHostToDevice));

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		
		// gpu work
		d_sumVectorElements << <blocksPerGrid, threadsPerBlock >> > (devAdjacencyMatrix1, devAdjacencyMatrix2, devAdjacencyMatrixDst, numberOfNodes_ * numberOfNodes_);
		
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float elapsedTime;
		cudaEventElapsedTime(&elapsedTime, start, stop);
		
		std::cout << std::endl;
		std::cout << "---> Duration in Graph addition: " << elapsedTime << " ms." << " <---" << std::endl;
		std::cout << std::endl;

		double *result = new double[numberOfNodes_ * numberOfNodes_];
		CUDA_ERR_CHECK(cudaMemcpy((void *)result, (const void *)devAdjacencyMatrixDst, totalBytes, cudaMemcpyDeviceToHost));

		CUDA_ERR_CHECK(cudaFree(devAdjacencyMatrix1));
		CUDA_ERR_CHECK(cudaFree(devAdjacencyMatrix2));
		CUDA_ERR_CHECK(cudaFree(devAdjacencyMatrixDst));

		return Graph((const double *)result, numberOfNodes_);
	}

	void Graph::generateGraph()
	{
		double *devAdjacencyMatrix;
		unsigned int totalBytes = sizeof(double) * numberOfNodes_ * numberOfNodes_;

		CUDA_ERR_CHECK(cudaMalloc((void**)&devAdjacencyMatrix, totalBytes));
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		// gpu work
		d_fillVectorRandomly << <blocksPerGrid, threadsPerBlock >> > (devAdjacencyMatrix, numberOfNodes_ * numberOfNodes_);

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float elapsedTime;
		cudaEventElapsedTime(&elapsedTime, start, stop);

		std::cout << std::endl;
		std::cout << "---> Duration in Graph generation with random numbers: " << elapsedTime << " ms." << " <---" << std::endl;

		CUDA_ERR_CHECK(cudaMemcpy(adjacencyMatrix_, devAdjacencyMatrix, totalBytes, cudaMemcpyDeviceToHost));
		//displayMatrix((const double *)adjacencyMatrix, numberOfNodes, numberOfNodes);

		CUDA_ERR_CHECK(cudaFree(devAdjacencyMatrix));
	}

	int Graph::getNumberOfNodes() const
	{
		return numberOfNodes_;
	}

	void Graph::display() const
	{
		displayMatrix((const double *)adjacencyMatrix_, numberOfNodes_, numberOfNodes_);
	}
}