#include "Graph.cuh"
#include "CudaHelper.cuh"
#include "VectorHelper.cuh"

namespace atspSolver
{
	Graph::Graph(int numberOfNodes) : numberOfNodes(numberOfNodes), adjacencyMatrix(new double[numberOfNodes*numberOfNodes])
	{
		Graph::generateGraph();
	}

	Graph::Graph(const double *adjacencyMatrix, int numberOfNodes) : numberOfNodes(numberOfNodes), adjacencyMatrix(new double[numberOfNodes*numberOfNodes])
	{
		memcpy(this->adjacencyMatrix, adjacencyMatrix, sizeof(double) * numberOfNodes * numberOfNodes);
	}

	Graph::~Graph()
	{
		delete[] adjacencyMatrix;
	}

	void Graph::copyElementsFromGraph(const Graph& graph)
	{
		double *devAdjacencyMatrixSrc, *devAdjacencyMatrixDst;
		unsigned int totalBytes = sizeof(double) * numberOfNodes * numberOfNodes;

		CUDA_ERR_CHECK(cudaMalloc((void**)&devAdjacencyMatrixSrc, totalBytes));
		CUDA_ERR_CHECK(cudaMalloc((void**)&devAdjacencyMatrixDst, totalBytes));
		CUDA_ERR_CHECK(cudaMemcpy((void *)devAdjacencyMatrixSrc, (const void *)graph.adjacencyMatrix, totalBytes, cudaMemcpyHostToDevice));
		d_copyVectorElements << <blocksPerGrid, threadsPerBlock >> > (devAdjacencyMatrixDst, devAdjacencyMatrixSrc, numberOfNodes * numberOfNodes);
		CUDA_ERR_CHECK(cudaMemcpy(adjacencyMatrix, devAdjacencyMatrixDst, totalBytes, cudaMemcpyDeviceToHost));

		CUDA_ERR_CHECK(cudaFree(devAdjacencyMatrixSrc));
		CUDA_ERR_CHECK(cudaFree(devAdjacencyMatrixDst));
	}

	Graph::Graph(const Graph& graph)
	{
		adjacencyMatrix = new double[graph.numberOfNodes * graph.numberOfNodes];
		numberOfNodes = graph.numberOfNodes;

		copyElementsFromGraph(graph);
	}

	Graph& Graph::operator=(const Graph& graph)
	{
		double *origAdjacencyMatrix = adjacencyMatrix;
		adjacencyMatrix = new double[graph.numberOfNodes * graph.numberOfNodes];
		numberOfNodes = graph.numberOfNodes;

		copyElementsFromGraph(graph);

		delete[] origAdjacencyMatrix;

		return *this;
	}

	double Graph::operator()(int row, int col) const
	{
		int index = col + row * numberOfNodes;
		return this->adjacencyMatrix[index];
	}

	double Graph::operator[](int index) const
	{
		return this->adjacencyMatrix[index];
	}

	Graph Graph::operator+(const Graph& graph) const
	{
		if (numberOfNodes != graph.numberOfNodes)
		{
			throw;
		}
		double *devAdjacencyMatrix1, *devAdjacencyMatrix2, *devAdjacencyMatrixDst;
		unsigned int totalBytes = sizeof(double) * numberOfNodes * numberOfNodes;

		CUDA_ERR_CHECK(cudaMalloc((void**)&devAdjacencyMatrix1, totalBytes));
		CUDA_ERR_CHECK(cudaMalloc((void**)&devAdjacencyMatrix2, totalBytes));
		CUDA_ERR_CHECK(cudaMalloc((void**)&devAdjacencyMatrixDst, totalBytes));
		CUDA_ERR_CHECK(cudaMemcpy((void *)devAdjacencyMatrix1, (const void *)graph.adjacencyMatrix, totalBytes, cudaMemcpyHostToDevice));
		CUDA_ERR_CHECK(cudaMemcpy((void *)devAdjacencyMatrix2, (const void *)adjacencyMatrix, totalBytes, cudaMemcpyHostToDevice));

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		
		// gpu work
		d_sumVectorElements << <blocksPerGrid, threadsPerBlock >> > (devAdjacencyMatrix1, devAdjacencyMatrix2, devAdjacencyMatrixDst, numberOfNodes * numberOfNodes);
		
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float elapsedTime;
		cudaEventElapsedTime(&elapsedTime, start, stop);
		
		std::cout << std::endl;
		std::cout << "---> Duration in Graph addition: " << elapsedTime << " ms." << " <---" << std::endl;
		std::cout << std::endl;

		double *result = new double[numberOfNodes * numberOfNodes];
		CUDA_ERR_CHECK(cudaMemcpy((void *)result, (const void *)devAdjacencyMatrixDst, totalBytes, cudaMemcpyDeviceToHost));

		CUDA_ERR_CHECK(cudaFree(devAdjacencyMatrix1));
		CUDA_ERR_CHECK(cudaFree(devAdjacencyMatrix2));
		CUDA_ERR_CHECK(cudaFree(devAdjacencyMatrixDst));

		return Graph((const double *)result, numberOfNodes);
	}

	void Graph::generateGraph()
	{
		double *devAdjacencyMatrix;
		unsigned int totalBytes = sizeof(double) * numberOfNodes * numberOfNodes;

		CUDA_ERR_CHECK(cudaMalloc((void**)&devAdjacencyMatrix, totalBytes));
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		// gpu work
		d_fillVectorRandomly << <blocksPerGrid, threadsPerBlock >> > (devAdjacencyMatrix, numberOfNodes * numberOfNodes);

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float elapsedTime;
		cudaEventElapsedTime(&elapsedTime, start, stop);

		std::cout << std::endl;
		std::cout << "---> Duration in Graph generation with random numbers: " << elapsedTime << " ms." << " <---" << std::endl;

		CUDA_ERR_CHECK(cudaMemcpy(adjacencyMatrix, devAdjacencyMatrix, totalBytes, cudaMemcpyDeviceToHost));
		//displayMatrix((const double *)adjacencyMatrix, numberOfNodes, numberOfNodes);

		CUDA_ERR_CHECK(cudaFree(devAdjacencyMatrix));
	}

	int Graph::getNumberOfNodes() const
	{
		return numberOfNodes;
	}

	void Graph::display() const
	{
		displayMatrix((const double *)adjacencyMatrix, numberOfNodes, numberOfNodes);
	}
}