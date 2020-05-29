#include <vector>
#include "Graph.cuh"
#include <memory>
#include <sstream>

namespace atspSolver
{
	struct fullCycle
	{
	public:
		fullCycle() : totalCost(0) {}
		fullCycle(std::vector<int> path, double totalCost) : totalCost(totalCost), path(path) {}
		std::vector<int> path;
		double totalCost;
		void display();
	};

	double calculateCost(const std::vector<int> &v, const Graph &graph);
	void permute(std::vector<int> a, int l, int r, const Graph& graph);
	fullCycle findOptimalPath(const Graph &graph);
}