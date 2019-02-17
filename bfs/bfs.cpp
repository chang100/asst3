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

void top_down_step(Graph g, int* distances, int curDistance, bool& finished)
{
  int nextDistance = curDistance + 1;
  #pragma omp parallel for
  for (int i = 0; i < num_nodes(g); i++)
  {
    if (distances[i] != curDistance) {
      continue;
    }

    for (const Vertex* v =outgoing_begin(g, i);
            v < outgoing_end(g,i); v++) {
      if (distances[*v] == NOT_VISITED_MARKER) {
        distances[*v] = nextDistance;
        finished = false;
      }
    }
  }
}

void top_down_step_reduced(Graph g, int* distances, int curDistance)
{
  int nextDistance = curDistance + 1;
  #pragma omp parallel for
  for (int i = 0; i < num_nodes(g); i++)
  {
    if (distances[i] != curDistance) {
      continue;
    }

    for (const Vertex* v =outgoing_begin(g, i);
            v < outgoing_end(g,i); v++) {
      if (distances[*v] == NOT_VISITED_MARKER) {
        distances[*v] = nextDistance;
      }
    }
  }
}

void bfs_top_down(Graph graph, solution* sol) {
  int numNodes = graph->num_nodes;
  memset(sol->distances, NOT_VISITED_MARKER, sizeof(int) * numNodes);
  sol->distances[ROOT_NODE_ID] = 0;
  bool finished = false;
  int curDistance = 0;

  while (!finished) {
    finished = true;
    top_down_step(graph, sol->distances, curDistance++, finished);
  }
}

void bottom_up_step(const Graph& g,
                   int* distances,
                   int curDistance,
                   bool& finished)
{
  int nextDistance = curDistance + 1;
  #pragma omp parallel for
  for (int i = 0; i < num_nodes(g); i++) {
    if (distances[i] == NOT_VISITED_MARKER) {
      for (const Vertex* v = incoming_begin(g, i); v < incoming_end(g, i); v++)
      {
        if (distances[*v] == curDistance) {
          distances[i] = nextDistance;
          finished = false;
          break;
        }
      }
    }
  }
}

void bottom_up_step_reduced(const Graph& g, int* distances, int curDistance)
{
  int nextDistance = curDistance + 1;
  #pragma omp parallel for
  for (int i = 0; i < num_nodes(g); i++) {
    if (distances[i] == NOT_VISITED_MARKER) {
      for (const Vertex* v = incoming_begin(g, i); v < incoming_end(g, i); v++)
      {
        if (distances[*v] == curDistance) {
          distances[i] = nextDistance;
        }
      }
    }
  }
}

void bfs_bottom_up(Graph graph, solution* sol)
{
  int numNodes = num_nodes(graph);
  memset(sol->distances, NOT_VISITED_MARKER, sizeof(int) * numNodes);

  // setup frontier with the root node
  sol->distances[ROOT_NODE_ID] = 0;
  int curDistance = 0;
  bool finished = false;

  while (!finished) {
    finished = true;
    bottom_up_step(graph, sol->distances, curDistance++, finished);
  }
}

void bfs_hybrid(Graph graph, solution* sol)
{
  const int numNodes = num_nodes(graph);
  const int THRESHOLD = (int) 0.01 * numNodes;
  memset(sol->distances, NOT_VISITED_MARKER, sizeof(int) * numNodes);

  // setup frontier with the root node
  sol->distances[ROOT_NODE_ID] = 0;
  int curDistance = 0;
  bool finished = false;
  int frontierSize = 1;

  while (frontierSize > 0) {
    if (frontierSize < THRESHOLD) {
      top_down_step_reduced(graph, sol->distances, curDistance++);
    }
    else {
      bottom_up_step_reduced(graph, sol->distances, curDistance++);
    }

    frontierSize = 0;
    #pragma omp parallel for shared(frontierSize)
    for (int i = 0; i < numNodes; i++) {
      if(sol->distances[i] == curDistance) {
        frontierSize++;
      }
    }
  }
}
