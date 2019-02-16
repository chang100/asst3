#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>
#include <vector>
#include <iostream>
#include <algorithm>

#include "../common/CycleTimer.h"
#include "../common/graph.h"


// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double* solution, double damping, double convergence)
{
  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
  double* score_old = new double[numNodes];

  std::vector<Vertex> sinkList;
  std::vector<Vertex> localSinkList;

  // Preprocess and find all nodes with no outgoing num_edges
  #pragma omp parallel shared(sinkList)
  {
    std::vector<Vertex> localSinkList;
    #pragma omp for
    for (int vi = 0; vi < numNodes; vi++) {
      //solution[vi] = equal_prob;
      score_old[vi] = equal_prob;

      if (outgoing_size(g, vi) == 0)
        localSinkList.push_back(vi);
    }
    #pragma omp critical
    {
      sinkList.insert(sinkList.end(), localSinkList.begin(), localSinkList.end());
    }
  }
  std::sort(sinkList.begin(), sinkList.end());

  while (true) {
    double sinkScore = 0.0;
    #pragma omp parallel for reduction (+:sinkScore)
    for (int i = 0; i < sinkList.size(); i++) {
      sinkScore = sinkScore + score_old[sinkList[i]] / numNodes;
    }
    sinkScore *= damping;

    #pragma omp parallel for
    for (Vertex vi = 0; vi < numNodes; vi++) {
      double sum = 0;
      for (const Vertex* v = incoming_begin(g, vi);
                    v != incoming_end(g, vi); v++) {
        sum += score_old[*v] / outgoing_size(g, *v);
      }

      solution[vi] = sinkScore + (damping * sum) + (1.0 - damping) / numNodes;
    }

    double sum = 0;
    #pragma omp parallel for reduction (+:sum)
    for (int vi = 0; vi < numNodes; vi++) {
      sum = sum + fabs(solution[vi] - score_old[vi]);
    }

    if (sum < convergence) {
      break;
    }

    #pragma omp parallel for
    for (int vi = 0; vi < numNodes; vi++) {
      score_old[vi] = solution[vi];
    }
  }

  /*
     CS149 students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes vj with no outgoing edges
                          { damping * score_old[vj] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */
}
