#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>
#include <iostream>
#include <algorithm>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1
//#define VERBOSE

void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

void vertex_set_init(vertex_set* list, int count) {
    list->max_vertices = count;
    list->vertices = (int*)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

void top_down_step(Graph g,
                   int* distances,
                   int curDistance,
                   int& frontierCount)
{
  int nextDistance = curDistance + 1;
  int frontierSize = 0;
  #pragma omp parallel for reduction(+:frontierSize)
  for (int i = 0; i < num_nodes(g); i++)
  {
    if (distances[i] != curDistance) {
      continue;
    }

    for (const Vertex* v =outgoing_begin(g, i);
            v < outgoing_end(g,i); v++) {
      if (distances[*v] == NOT_VISITED_MARKER) {
        distances[*v] = nextDistance;
        frontierSize += 1;
      }
    }
  }

  frontierCount = frontierSize;
}

void bfs_top_down(Graph graph, solution* sol) {
  int numNodes = graph->num_nodes;
  memset(sol->distances, NOT_VISITED_MARKER, sizeof(int) * numNodes);
  sol->distances[ROOT_NODE_ID] = 0;
  int frontierSize = 1;
  int curDistance = 0;

  while (frontierSize > 0) {
    top_down_step(graph, sol->distances, curDistance++, frontierSize);
  }
}

int bottom_up_step(const Graph& g,
                   int* distances,
                   bool*& frontier,
                   bool*& newFrontier,
                   int curDistance,
                   int& frontierCount)
{
  int nextDistance = curDistance + 1;
  int frontierSize = 0;
  memset(newFrontier, 0, num_nodes(g));
  //#pragma omp parallel
  #pragma omp parallel for reduction(+:frontierSize)
  for (int i = 0; i < num_nodes(g); i++) {
    if (distances[i] == NOT_VISITED_MARKER) {
      bool found = false;
      const Vertex* start = incoming_begin(g, i);
      const Vertex* end = incoming_end(g, i);
      for (const Vertex* v = start; v < end; v++)
      {
        if (frontier[*v]) {
          distances[i] = nextDistance;
          newFrontier[i] = true;
          frontierSize += 1;
          break;
        }
      }
    }
  }

  bool* tmp = frontier;
  frontier = newFrontier;
  newFrontier = tmp;
  frontierCount = frontierSize;
}

void bfs_bottom_up(Graph graph, solution* sol)
{
  int numNodes = num_nodes(graph);
  int* start = sol->distances;
  int* end = start + numNodes;
  memset(start, NOT_VISITED_MARKER, sizeof(int) * numNodes);

  bool* frontier = new bool[numNodes];
  bool* newFrontier = new bool[numNodes];
  memset(frontier, 0, numNodes);

  // setup frontier with the root node
  sol->distances[ROOT_NODE_ID] = 0;
  int curDistance = 0;
  bool finished = false;
  int frontierSize = 1;
  frontier[ROOT_NODE_ID] = true;

  while (frontierSize > 0) {
    finished = true;
    bottom_up_step(graph,
                   sol->distances,
                   frontier,
                   newFrontier,
                   curDistance++,
                   frontierSize);
  }

  delete frontier;
  delete newFrontier;
}

void convertToBinary(bool* frontier,
                     int* distances,
                     int curDistance,
                     int numNodes)
{
  memset(frontier, false, numNodes);
  #pragma omp parallel for
  for (int i = 0; i < numNodes; i++) {
    if (distances[i] == curDistance - 1) {
      frontier[i] = true;
    }
  }
}

void bfs_hybrid(Graph graph, solution* sol)
{
  const int numNodes = num_nodes(graph);
  const int THRESHOLD = (int) 1.0 * numNodes;
  memset(sol->distances, NOT_VISITED_MARKER, sizeof(int) * numNodes);

  bool* frontier = new bool[numNodes];
  bool* newFrontier = new bool[numNodes];
  memset(frontier, false, numNodes);
  memset(newFrontier, false, numNodes);

  // setup frontier with the root node
  sol->distances[ROOT_NODE_ID] = 0;
  frontier[ROOT_NODE_ID] = true;
  int curDistance = 0;
  bool finished = false;
  int frontierSize = 1;

  bool binaryRepresentation = true;

  while (frontierSize > 0) {
    if (frontierSize < THRESHOLD) {
      binaryRepresentation = false;
      top_down_step(graph, sol->distances, curDistance++, frontierSize);
    }
    else {
      if (!binaryRepresentation) {
        convertToBinary(frontier, sol->distances, curDistance, numNodes);
      }
      bottom_up_step(graph,
                     sol->distances,
                     frontier,
                     newFrontier,
                     curDistance++,
                     frontierSize);
    }
  }
}
