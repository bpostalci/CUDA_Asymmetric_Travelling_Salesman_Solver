// Berat Postalcioglu
/* OUTPUT

	Asymmetric Traveling Salesman Problem Solver with random weighted 5 Nodes
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	---> Duration in Graph generation with random numbers: 48.7383 ms. <---

	=== g1 adjacency matrix ===
	m[0]: 896, m[1]: 857, m[2]: 53, m[3]: 107, m[4]: 987,
	m[5]: 570, m[6]: 115, m[7]: 713, m[8]: 708, m[9]: 757,
	m[10]: 125, m[11]: 286, m[12]: 385, m[13]: 582, m[14]: 786,
	m[15]: 539, m[16]: 564, m[17]: 333, m[18]: 198, m[19]: 617,
	m[20]: 159, m[21]: 640, m[22]: 86, m[23]: 413, m[24]: 973

	=== g2 adjacency matrix ===
	m[0]: 896, m[1]: 857, m[2]: 53, m[3]: 107, m[4]: 987,
	m[5]: 570, m[6]: 115, m[7]: 713, m[8]: 708, m[9]: 757,
	m[10]: 125, m[11]: 286, m[12]: 385, m[13]: 582, m[14]: 786,
	m[15]: 539, m[16]: 564, m[17]: 333, m[18]: 198, m[19]: 617,
	m[20]: 159, m[21]: 640, m[22]: 86, m[23]: 413, m[24]: 973


	---> Duration in Graph addition: 0.01024 ms. <---

	=== g3 adjacency matrix = (g1 + g2) ===
	m[0]: 1792, m[1]: 1714, m[2]: 106, m[3]: 214, m[4]: 1974,
	m[5]: 1140, m[6]: 230, m[7]: 1426, m[8]: 1416, m[9]: 1514,
	m[10]: 250, m[11]: 572, m[12]: 770, m[13]: 1164, m[14]: 1572,
	m[15]: 1078, m[16]: 1128, m[17]: 666, m[18]: 396, m[19]: 1234,
	m[20]: 318, m[21]: 1280, m[22]: 172, m[23]: 826, m[24]: 1946

	=== Optimal Path Found ===
	0 -> 3 -> 1 -> 4 -> 2 -> 0  |  cost: 3278

*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include "Graph.cuh"
#include "Solver.cuh"

using namespace std;
using namespace atspSolver;

const int NumberOfNodes = 5;

int main()
{
	cout << "Asymmetric Traveling Salesman Problem Solver with random weighted 5 Nodes" << endl;
	cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;

	Graph g1(NumberOfNodes);
	Graph g2 = g1;

	cout << endl;
	cout << "=== g1 adjacency matrix === " << endl;
	g1.display();
	cout << endl;
	cout << "=== g2 adjacency matrix === " << endl;
	g2.display();

	cout << endl;
	Graph g3 = g1 + g2;
	cout << "=== g3 adjacency matrix = (g1 + g2) === " << endl;
	g3.display();
	
	fullCycle optimalPath = findOptimalPath(g3);
	cout << endl;
	cout << "=== Optimal Path Found ===" << endl;
	optimalPath.display();

	return 0;
}