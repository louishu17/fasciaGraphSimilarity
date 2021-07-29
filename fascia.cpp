// Copyright (c) 2013, The Pennsylvania State University.
// All rights reserved.
// 
// See COPYING for license.

using namespace std;

#include <stdio.h>
#include <cstdlib>
#include <assert.h>
#include <fstream>
#include <math.h>
#include <sys/time.h>
#include <vector>
#include <iostream>
#include <sys/stat.h>
#include <cstring>
#include <unistd.h>
#include <climits>
#include <numeric>
#include <unordered_set>
#include <algorithm>
#include <sstream>

#include <stdlib.h>
#include <random>
#include <string>
#include <tuple>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "fascia.h"
#include "graph.hpp"
#include "util.hpp"
#include "output.hpp"
#include "dynamic_table.hpp"
#include "dynamic_table_array.hpp"
#include "partitioner.hpp"
#if SIMPLE
  #include "colorcount_simple.hpp"
#else
  #include "colorcount.hpp"
#endif

bool timing = false;

void print_info_short(char* name)
{
  printf("\nTo run: %s [-g graphfile] [-t template || -b batchfile] [options]\n", name);
  printf("Help: %s -h\n\n", name);

  exit(0);
}

void print_info(char* name)
{
  printf("\nTo run: %s [-g graphfile] [-t template || -b batchfile] [options]\n\n", name);

  printf("\tgraphfile = \n");
  printf("\t\tn\n");
  printf("\t\tm\n");
  printf("\t\tv0 v1\n");
  printf("\t\tv0 v2\n");
  printf("\t\t...\n");
  printf("\t\t(zero indexed)\n\n");

  printf("\tgraphfile (if labeled) = \n");
  printf("\t\tn\n");
  printf("\t\tm\n");
  printf("\t\tlabel_v0\n");
  printf("\t\tlabel_v1\n");
  printf("\t\t...\n");
  printf("\t\tv0 v1\n");
  printf("\t\tv0 v2\n");
  printf("\t\t...\n");
  printf("\t\t(zero indexed)\n\n"); 

  printf("\ttemplate =\n");
  printf("\t\tsame format as graphfile\n\n");

  printf("\tbatchfile =\n");
  printf("\t\ttemplate1\n");
  printf("\t\ttemplate2\n");
  printf("\t\t...\n");
  printf("\t\t(must supply only one of template file or batchfile)\n\n");

  printf("\toptions = \n");
  printf("\t\t-m  [#], compute counts for motifs of size #\n");
  printf("\t\t-o  Use outerloop parallelization\n");
  printf("\t\t-l  Graph and template are labeled\n");
  printf("\t\t-i  [# iterations], default: 1\n");
  printf("\t\t-c  Output per-vertex counts to [template].vert\n");
  printf("\t\t-d  Output graphlet degree distribution to [template].gdd\n");
  printf("\t\t-a  Do not calculate automorphism of template\n");
  printf("\t\t\t(recommended when template size > 10)\n");
  printf("\t\t-r  Report runtime\n");
  printf("\t\t-v  Verbose output\n");
  printf("\t\t-h  Print this\n\n");

  exit(0);
}

void read_in_graph(Graph& g, char* graph_file, bool labeled,
  int*& srcs_g, int*& dsts_g, int*& labels_g)
{
  ifstream file_g;
  string line;
  
  file_g.open(graph_file);
  
  int n_g;
  int m_g;    

  getline(file_g, line);
  n_g = atoi(line.c_str());
  getline(file_g, line);
  m_g = atoi(line.c_str());
  
  srcs_g = new int[m_g];
  dsts_g = new int[m_g];

  if (labeled)
  {
    labels_g = new int[n_g];
    for (int i = 0; i < n_g; ++i)
    {
      getline(file_g, line);
      labels_g[i] = atoi(line.c_str());
    }
  }
  else
  {
    labels_g = NULL;
  }
  
  for (int  i = 0; i < m_g; ++i)
  {
    getline(file_g, line, ' ');   
    srcs_g[i] = atoi(line.c_str());
    getline(file_g, line);  
    dsts_g[i] = atoi(line.c_str());
  } 
  file_g.close();
  
  g.init(n_g, m_g, srcs_g, dsts_g);
  //print_my_graph(g);
}

double run_single(char* graph_file, char* template_file, bool labeled,
                bool do_vert, bool do_gdd,
                int iterations, 
                bool do_outerloop, bool calc_auto, bool verbose, bool random_graphs, float p, bool main, bool isCentered)
{
  Graph g;
  Graph t;
  int* srcs_g;
  int* dsts_g;
  int* labels_g;
  int* srcs_t;
  int* dsts_t;
  int* labels_t;
  char* vert_file = new char[1024];
  char* gdd_file = new char[1024];

  if (do_vert) {
    strcat(vert_file, template_file);
    strcat(vert_file, ".vert");
  }
  if (do_gdd) {
    strcat(gdd_file, template_file);
    strcat(gdd_file, ".gdd");
  }

  read_in_graph(g, graph_file, labeled, srcs_g, dsts_g, labels_g);
  read_in_graph(t, template_file, labeled, srcs_t, dsts_t, labels_t);

  double elt = 0.0;
  if ((timing || verbose) && main) {
    elt = timer();
  }
  double full_count = 0.0;  
  if (do_outerloop)
  {
    int num_threads = omp_get_max_threads();
    int iter = ceil( (double)iterations / (double)num_threads + 0.5);
    
    colorcount* graph_count = new colorcount[num_threads];
    for (int tid = 0; tid < num_threads; ++tid) {
      graph_count[tid].init(g, labels_g, labeled, 
                            calc_auto, do_gdd, do_vert, verbose);
    }

    double** vert_counts;
    if (do_gdd || do_vert)
      vert_counts = new double*[num_threads];

#pragma omp parallel reduction(+:full_count)
{
    int tid = omp_get_thread_num();
    full_count += graph_count[tid].do_full_count(&t, labels_t, iter, random_graphs, p, isCentered);
    if (do_gdd || do_vert)
      vert_counts[tid] = graph_count[tid].get_vert_counts();
}   
    full_count /= (double)num_threads;
    if (do_gdd || do_vert)
    {
      output out(vert_counts, num_threads, g.num_vertices());
      if (do_gdd) {
        out.output_gdd(gdd_file);
        free(gdd_file);
      } 
      if (do_vert) {        
        out.output_verts(vert_file);
        free(vert_file);
      }
    }
  }
  else
  {
    colorcount graph_count;
    graph_count.init(g, labels_g, labeled, 
                      calc_auto, do_gdd, do_vert, verbose);
    full_count += graph_count.do_full_count(&t, labels_t, iterations, random_graphs, p, isCentered);

    if (do_gdd || do_vert)
    {
      double* vert_counts = graph_count.get_vert_counts();
      output out(vert_counts, g.num_vertices());
      if (do_gdd)
      {
        out.output_gdd(gdd_file);
        free(gdd_file);
      }
      if (do_vert)
      {
        out.output_verts(vert_file);
        free(vert_file);
      }
    }
  }

  printf("%e", full_count);

  if ((timing || verbose) && main) {
    elt = timer() - elt;
    printf("Total time:\n\t%9.6lf seconds\n", elt);
  }

  delete [] srcs_g;
  delete [] dsts_g;
  delete [] labels_g;
  delete [] srcs_t;
  delete [] dsts_t;
  delete [] labels_t;
  
  return full_count;

}


void run_batch(char* graph_file, char* batch_file, bool labeled,
                bool do_vert, bool do_gdd,
                int iterations, 
                bool do_outerloop, bool calc_auto, bool verbose, bool random_graphs, float p, bool main, bool isCentered)
{
  Graph g;
  Graph t;
  int* srcs_g;
  int* dsts_g;
  int* labels_g;
  int* srcs_t;
  int* dsts_t;
  int* labels_t;
  char* vert_file;
  char* gdd_file;

  std::vector<double> full_count_arr;

  read_in_graph(g, graph_file, labeled, srcs_g, dsts_g, labels_g);

  double elt = 0.0;
  if ((timing || verbose) && main) {
    elt = timer();
  }

  ifstream if_batch;
  string line;
  if_batch.open(batch_file);
  while (getline(if_batch, line))
  {   
    char* template_file = strdup(line.c_str());
    read_in_graph(t, template_file, labeled, srcs_t, dsts_t, labels_t);

    double full_count = 0.0;
    if (do_outerloop)
    {
      int num_threads = omp_get_max_threads();
      int iter = ceil( (double)iterations / (double)num_threads + 0.5);
      
      colorcount* graph_count = new colorcount[num_threads];
      for (int i = 0; i < num_threads; ++i) {
        graph_count[i].init(g, labels_g, labeled, 
                            calc_auto, do_gdd, do_vert, verbose);
      }

    
      double** vert_counts;
      if (do_gdd || do_vert)
        vert_counts = new double*[num_threads];

#pragma omp parallel reduction(+:full_count)
{
      int tid = omp_get_thread_num();
      full_count += graph_count[tid].do_full_count(&t, labels_t, iter, random_graphs, p, isCentered);
      if (do_gdd || do_vert)
        vert_counts[tid] = graph_count[tid].get_vert_counts();
}   
      full_count /= (double)num_threads;
      if (do_gdd || do_vert)
      {
        output out(vert_counts, num_threads, g.num_vertices());
        if (do_gdd) {
          gdd_file = strdup(template_file);
          strcat(gdd_file, ".gdd");
          out.output_gdd(gdd_file);
          free(gdd_file);
        }
        if (do_vert) {
          vert_file = strdup(template_file);
          strcat(vert_file, ".vert");
          out.output_verts(vert_file);
          free(vert_file);
        }
      }
    }
    else
    {
      colorcount graph_count;
      graph_count.init(g, labels_g, labeled, 
                        calc_auto, do_gdd, do_vert, verbose);
      full_count += graph_count.do_full_count(&t, labels_t, iterations, random_graphs, p, isCentered);
    }

    // printf("%e\n", full_count);  
    // check count_automorphissms
    // printf("num of automorphisms: %d\n", count_automorphisms(t));
    full_count_arr.push_back(full_count * sqrt(count_automorphisms(t)));


    delete [] srcs_t;
    delete [] dsts_t;
    delete [] labels_t;
    delete [] template_file;
  }

  if_batch.close();

if ((timing || verbose) && main) {
  elt = timer() - elt;
  printf("Total time:\n\t%9.6lf seconds\n", elt);
}

  delete [] srcs_g;
  delete [] dsts_g;
  delete [] labels_g;
}

int main(int argc, char** argv)
{

  char * graph_fileA = strdup("small_fb/Caltech36.txt");
  char * batch_file = strdup("motif/graphs_n5_3/batchfile");
  
  for (int i = 1; i <= 10000; ++i) {
    run_batch(graph_fileA, batch_file, false, false, false, 1, false, true, false, false, 0, false, true);
  }

}