#include <iostream>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Graph.cuh"
#include "Solver.cuh"

using namespace std;

namespace atspSolver
{
	void fullCycle::display()
	{
		std::stringstream stream;
		int pathSize = path.size();

		for (int i = 0; i < pathSize; i++)
		{
			stream << path[i];
			if (i != pathSize - 1) stream << " -> ";
		}
		stream << "  |  cost: " << totalCost;
		std::cout << stream.str() << std::endl;
	}

	double calculateCost(const std::vector<int> &v, const Graph &graph)
	{
		double result = 0;
		int vectorSize = v.size();
		int numberOfNodes = graph.getNumberOfNodes();

		for (int i = 0; (i + 1) < vectorSize; i++)
		{
			// v[i] = row, v[i + 1] = column
			int graphIndex = v[i + 1] + v[i] * numberOfNodes;
			result += graph[graphIndex];
		}

		return result;
	}

	std::vector<fullCycle> fullCycles;
	void permute(vector<int> a, int l, int r, const Graph& graph)
	{
		if (l == r) {
			a.push_back(a[0]);
			double cost = calculateCost(a, graph);
			fullCycles.push_back(fullCycle(a, cost));
		}
		else
		{
			for (int i = l; i <= r; i++)
			{
				swap(a[l], a[i]);

				// Recursion 
				permute(a, l + 1, r, graph);

				// Backtrack  
				swap(a[l], a[i]);
			}
		}
	}

	fullCycle findOptimalPath(const Graph &graph)
	{
		std::vector<int> v;
		for (int i = 0; i < graph.getNumberOfNodes(); i++)
		{
			v.push_back(i);
		}

		permute(v, 0, v.size() - 1, graph);

		fullCycle result = fullCycles[0];
		for (int i = 0; i < fullCycles.size(); i++)
		{
			if (fullCycles[i].totalCost < result.totalCost)
			{
				result = fullCycles[i];
			}
		}

		return result;
	}
}