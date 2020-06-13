#pragma once

namespace atspSolver
{
	// Graph implementation with adjacency matrix
	class Graph;
	// to make Graph class final
	class Final
	{
	private:
		Final() {};
		friend class Graph;
	};

	class Graph : virtual Final
	{
	public:
		Graph(int numberOfNodes = 1);
		Graph(const double *adjacencyMatrix, int numberOfNodes = 1);
		~Graph();
		Graph(const Graph& graph);
		Graph& operator=(const Graph& graph);
		double operator()(int row, int col) const;
		double operator[](int index) const;
		Graph operator+(const Graph& graph) const;
		int getNumberOfNodes() const;
		void display() const;
	private:
		int numberOfNodes_;
		double* adjacencyMatrix_;
		void generateGraph();
		void copyElementsFromGraph(const Graph& graph);
	};
}