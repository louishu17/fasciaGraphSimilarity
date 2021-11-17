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
    full_count += graph_count[tid].do_full_count(&t, labels_t, iter, random_graphs, p, isCentered, 23);
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
    full_count += graph_count.do_full_count(&t, labels_t, iterations, random_graphs, p, isCentered, 23);

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

  // printf("%e", full_count);

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


std::vector<double> run_batch(char* graph_file, char* batch_file, bool labeled,
                bool do_vert, bool do_gdd,
                int iterations, 
                bool do_outerloop, bool calc_auto, bool verbose, bool random_graphs, float p, bool main, bool isCentered, int colorKey)
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
      full_count += graph_count[tid].do_full_count(&t, labels_t, iter, random_graphs, p, isCentered, colorKey);
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
      full_count += graph_count.do_full_count(&t, labels_t, iterations, random_graphs, p, isCentered, colorKey);
    }

    // printf("\n");

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

  return full_count_arr;
}



std::vector<double> run_motif(char* graph_file, int motif, 
                bool do_vert, bool do_gdd, 
                int iterations, 
                bool do_outerloop, bool calc_auto, bool verbose, bool random_graphs, float p, bool main, bool isCentered, int colorKey)
{
  char* motif_batchfile = NULL;

  switch(motif)
  {
    case(3):
      motif_batchfile = strdup("motif/graphs_n3_1/batchfile");
      break;
    case(4):
      motif_batchfile = strdup("motif/graphs_n4_2/batchfile");
      break;
    case(5):
      motif_batchfile = strdup("motif/graphs_n5_3/batchfile");
      break;
    case(6):
      motif_batchfile = strdup("motif/graphs_n6_6/batchfile");
      break;
    case(7):
      motif_batchfile = strdup("motif/graphs_n7_11/batchfile");
      break;
    case(8):
      motif_batchfile = strdup("motif/graphs_n8_23/batchfile");
      break;
    case(9):
      motif_batchfile = strdup("motif/graphs_n9_47/batchfile");
      break;
    case(10):
      motif_batchfile = strdup("motif/graphs_n10_106/batchfile");
      break;
    default:
      break;
  }

  return run_batch(graph_file, motif_batchfile, false,
            do_vert, do_gdd,
            iterations, 
            do_outerloop, calc_auto, verbose, random_graphs, p, main, isCentered, colorKey);
}

double run_algorithm2(char* graph_fileA, char* graph_fileB, int motif, bool labeled,
                bool do_vert, bool do_gdd, 
                int iterations, 
                bool do_outerloop, bool calc_auto, bool verbose, bool random_graphs, float p, bool main, bool isCentered) {
  Graph gA;
  Graph gB;
  Graph t;
  int* srcs_gA;
  int* dsts_gA;
  int* labels_gA;
  int* srcs_t;
  int* dsts_t;
  int* srcs_gB;
  int* dsts_gB;
  int* labels_gB;
  int* labels_t;
  char* vert_file;
  char* gdd_file;

  char* motif_batchfile = NULL;

  switch(motif)
  {
    case(3):
      motif_batchfile = strdup("motif/graphs_n3_1/batchfile");
      break;
    case(4):
      motif_batchfile = strdup("motif/graphs_n4_2/batchfile");
      break;
    case(5):
      motif_batchfile = strdup("motif/graphs_n5_3/batchfile");
      break;
    case(6):
      motif_batchfile = strdup("motif/graphs_n6_6/batchfile");
      break;
    case(7):
      motif_batchfile = strdup("motif/graphs_n7_11/batchfile");
      break;
    case(8):
      motif_batchfile = strdup("motif/graphs_n8_23/batchfile");
      break;
    case(9):
      motif_batchfile = strdup("motif/graphs_n9_47/batchfile");
      break;
    case(10):
      motif_batchfile = strdup("motif/graphs_n10_106/batchfile");
      break;
    default:
      break;
  }

  double final_count = 0.0;
  int colorKey_A = rand();
  int colorKey_B = rand();


  read_in_graph(gA, graph_fileA, labeled, srcs_gA, dsts_gA, labels_gA);
  read_in_graph(gB, graph_fileB, labeled, srcs_gB, dsts_gB, labels_gB);

  double elt = 0.0;
  if ((timing || verbose) && main) {
    elt = timer();
  }

  ifstream if_batch;
  string line;
  if_batch.open(motif_batchfile);
  while (getline(if_batch, line))
  {   
    char* template_file = strdup(line.c_str());
    read_in_graph(t, template_file, labeled, srcs_t, dsts_t, labels_t);

    double full_countA = 0.0;
    double full_countB = 0.0;

    

    if (do_outerloop)
    {
      int num_threads = omp_get_max_threads();
      int iter = ceil( (double)iterations / (double)num_threads + 0.5);
      
      colorcount* graph_countA = new colorcount[num_threads];
      colorcount* graph_countB = new colorcount[num_threads];
      for (int i = 0; i < num_threads; ++i) {
        graph_countA[i].init(gA, labels_gA, labeled, 
                            calc_auto, do_gdd, do_vert, verbose);
        graph_countB[i].init(gB, labels_gB, labeled, 
                            calc_auto, do_gdd, do_vert, verbose);
      }

    
      double** vert_counts;
      if (do_gdd || do_vert)
        vert_counts = new double*[num_threads];

#pragma omp parallel reduction(+:full_countA)
{
      int tid = omp_get_thread_num();
      full_countA += graph_countA[tid].do_full_count(&t, labels_t, iter, random_graphs, p, isCentered, colorKey_A);


      if (do_gdd || do_vert)
        vert_counts[tid] = graph_countA[tid].get_vert_counts();
}
#pragma omp parallel reduction(+:full_countB)
{
      int tid = omp_get_thread_num();
      full_countB += graph_countB[tid].do_full_count(&t, labels_t, iter, random_graphs, p, isCentered, colorKey_B);
      
      if (do_gdd || do_vert)
        vert_counts[tid] = graph_countB[tid].get_vert_counts();
}

      full_countA /= (double)num_threads;
      full_countB /= (double)num_threads;
      if (do_gdd || do_vert)
      {
        output out(vert_counts, num_threads, gA.num_vertices());
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
      colorcount graph_countA;
      colorcount graph_countB;
      
      // printf("fullcountA: %e\n", full_countA);
      // printf("fullcountB: %e\n", full_countB);

      graph_countA.init(gA, labels_gA, labeled, 
                        calc_auto, do_gdd, do_vert, verbose);
      graph_countB.init(gB, labels_gB, labeled, 
                        calc_auto, do_gdd, do_vert, verbose);
      full_countA += graph_countA.do_full_count(&t, labels_t, iterations, random_graphs, p, isCentered, colorKey_A);
      full_countB += graph_countB.do_full_count(&t, labels_t, iterations, random_graphs, p, isCentered, colorKey_B);
    }

    // printf("fullcountA: %e\n", full_countA);
    // printf("fullcountB: %e\n", full_countB);
 
    // check count_automorphissms
    // printf("num of automorphisms: %d\n", count_automorphisms(t));
    final_count += full_countA * full_countB * count_automorphisms(t);
    //printf("%e\n", final_count);


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

  delete [] srcs_gA;
  delete [] dsts_gA;
  delete [] labels_gA;
  delete [] srcs_gB;
  delete [] dsts_gB;
  delete [] labels_gB;

  // printf("%e\n", final_count);

  return final_count;
                  

}

double run_compare_graphs(char* graph_fileA, char* graph_fileB, int motif, 
                bool do_vert, bool do_gdd, 
                int iterations, 
                bool do_outerloop, bool calc_auto, bool verbose, bool random_graphs, float p, bool main, bool isCentered)
{

  double elt;
  if (timing && main) {
    elt = timer();
  }

  int colorKey = rand();
  
  std::vector<double> a = run_motif(graph_fileA, motif, do_vert, do_gdd, iterations, do_outerloop, calc_auto, verbose, random_graphs, p, false, isCentered, colorKey);
  std::vector<double> b = run_motif(graph_fileB, motif, do_vert, do_gdd, iterations, do_outerloop, calc_auto, verbose, random_graphs, p, false, isCentered, colorKey);

  double stat = std::inner_product(std::begin(a), std::end(a), std::begin(b), 0.0);

  std::vector<double>().swap(a);
  std::vector<double>().swap(b);

  if (timing && main) {
    elt = timer() - elt;
    printf("Timing: %f\n", elt);
  }

  if (main) {
    printf("%e", stat);
  }

  return stat;
}

void generate_graph(int n, float p, char filename[100])
{
// generates erdos-renyi graph with n nodes, probability p, saves it to a file with number file_num

    ofstream file(filename);

    int count = 0;
    vector <tuple<int, int>> edges;
    tuple<int, int> tup;

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            float r = ((float) rand() / (RAND_MAX));
            if (r < p) {
                ++count;
                tup = make_tuple(i, j);
                edges.push_back(tup);
            }
        }
    }

    file << n << '\n';
    file << count;

    for (auto it = edges.begin(); it != edges.end(); ++it) {
        file << '\n' << get<0>(*it) << ' ' << get<1>(*it);
    }

    file.close();

}

double calculateExpectedValue(int n, float p, int K, int aut) {
    // printf("%d %f %d %d", n, p, K, aut);
    double fac = 1;

    for(int i = n; i > n - K - 1; i--) {
        fac *= i;
    }

    fac /= aut;
    fac *= pow(p,K);
    return fac;
}

void sim1(int m, int iterations, float p, bool random_graphs, bool isCentered) {

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;
    //runs simulation 1
    Graph t;
    int* srcs_t;
    int* dsts_t;
    int* labels_t;

    char tree_file[] = "template.graph";

    read_in_graph(t, tree_file, false, srcs_t, dsts_t, labels_t);
    int aut = count_automorphisms(t);

    int K = 6;
    double r = factorial(K+1)/pow(K+1, K+1);
    int tconst = floor(1/pow(r,2));
    

    std::vector<int> m_sizes;

    for (int i = 1; i <= 20; ++i) {
      m_sizes.push_back(i * 50);
    }

    //m_sizes.push_back(100);
    //m_sizes.push_back(500);
    //m_sizes.push_back(1000);
    //m_sizes.push_back(10000);  

    std::vector<double> fin_counts;
    std::vector<double> expected_counts;
    double val;
    double expectedVal;
    double total_time;
    double avg_time;

    cout << "\n";

    for (const int& i : m_sizes) {
      val = 0;
      expectedVal = 0;
      // cout << i << "\n";
      // expectedVal = calculateExpectedValue(i, p, K, aut);
      // cout << "\nExpected Value: " << expectedVal << "\n";
      // cout << "[";
      for(int j = 0; j < m; ++j) {
        char graph_file [100];
        sprintf (graph_file, "graphs/%d.txt", i);
        generate_graph(i, p, graph_file);
        auto a1 = high_resolution_clock::now();
        val = val + run_single(graph_file, tree_file, false, false, false, iterations, false, true, false, random_graphs, p, false, isCentered);

        auto a2 = high_resolution_clock::now();

        auto ms_int = duration_cast<milliseconds>(a2-a1);

        duration<double, std::milli> ms_double = a2 - a1;

        total_time += ms_double.count();
        // cout << ", ";
        cout.flush();   
      }
      val = val / m;
      avg_time = total_time / m;
      fin_counts.push_back(val);
      // cout<<'\b';
      // cout<<'\b';
      // cout<<"]\n";
      cout << avg_time << "\n";
      cout.flush();      
      // expected_counts.push_back(expectedVal * r);
    }

    // cout << '[';
    // for (const double& d : fin_counts) {
    //   cout << d << ',';
    // }
    // cout << "]\n";

    // cout << '[';
    // for (const double& d : expected_counts) {
    //   cout << d << ',';
    // }
    // cout << "]\n";
    cout.flush();
}

void select_edges(float s, char in [100], char out [100]) {

    int n;
    int num_edges;

    ifstream og_erd_ren(in);
    og_erd_ren >> n;
    og_erd_ren >> num_edges;

    int count = 0;
    vector <tuple<int, int>> edges;
    int edge_1;
    int edge_2;
    tuple<int, int> tup;

    while (og_erd_ren >> edge_1 >> edge_2) {
        float r = ((float) rand() / (RAND_MAX));
        if (r < s) {
            ++count;
            tup = make_tuple(edge_1, edge_2);
            edges.push_back(tup);
            }
    }
    og_erd_ren.close();

    ofstream out_file(out);

    out_file << n << '\n';
    out_file << count;

    for (auto it = edges.begin(); it != edges.end(); ++it) {
        out_file << '\n' << get<0>(*it) << ' ' << get<1>(*it);
    }
    out_file.close();

}

void generate_corr_graphs(int n, float p, float s, int m_rep) {
    const char folder [] = "sim2_corr/";
    char in [100];
    sprintf(in, "%s%d_og.txt", folder, m_rep);
    char graphA [100];
    sprintf(graphA, "%s%d_%dA_%.5f_%.5f_corr.txt", folder, m_rep, n, p, s);
    char graphB [100];
    sprintf(graphB, "%s%d_%dB_%.5f_%.5f_corr.txt", folder, m_rep, n, p, s);
    
    generate_graph(n, p, in);
    select_edges(s, in, graphA);
    select_edges(s, in, graphB);
}

void generate_ind_graphs(int n, float p, float s, int m_rep) {
    const char folder [] = "sim2_ind/";
    char graphA [100];
    sprintf(graphA, "%s%d_%dA_%.5f_%.5f_ind.txt", folder, m_rep, n, p, s);
    char graphB [100];
    sprintf(graphB, "%s%d_%dB_%.5f_%.5f_ind.txt", folder, m_rep, n, p, s);
    
    generate_graph(n, p, graphA);
    generate_graph(n, p, graphB);

}

void generate_both_graphs(int n, float p, float s, int m) {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;


    // double r = (double) factorial(tree_len+1) / pow(tree_len+1, tree_len+1);
    // int t = floor(1/ pow(r,2));
    
    auto a1 = high_resolution_clock::now();

    for(int m_rep = 1; m_rep < m+1; ++m_rep) {
      generate_corr_graphs(n, p, s, m_rep);
      generate_ind_graphs(n, p, s, m_rep);
    }


    auto a2 = high_resolution_clock::now();

    auto ms_int = duration_cast<milliseconds>(a2-a1);

    duration<double, std::milli> ms_double = a2 - a1;

    std::cout << "\nTime to generate graphs: " << ms_double.count() << "ms";

}

void sim2(char* graph_fileA, char* graph_fileB, int n, float p, float s, int klow, int khigh, int m, int iterations, bool isCentered, bool Sophie) {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;


    for(int k = klow+1; k < khigh+2; ++k) {
      auto t1 = high_resolution_clock::now();

      // std::cout << "\n";

      double count = 0.0;

// #pragma omp parallel reduction(+:count)
// {
      if(Sophie) {
        for(int j = 0; j < iterations; ++j) {
          // printf("Total: %e\n", count);
          count += run_algorithm2(graph_fileA, graph_fileB, k, false, false, false, 1, false, true, false, true, p*s, true, isCentered);
        }
        count /= iterations;
      }
      else {
        count = run_compare_graphs(graph_fileA, graph_fileB, k, false, false, iterations, false, true, false, true, p*s, false, isCentered);
      }

// }

      printf("%e, ", count);


      auto t2 = high_resolution_clock::now();

      auto ms_int = duration_cast<milliseconds>(t2-t1);

      duration<double, std::milli> ms_double = t2 - t1;

      // std::cout << "\n" << ms_double.count() << "ms";
      std::cout.flush();
    }



}

std::unordered_set<int> pickSet(int N, int k, std::mt19937& gen)
{
    std::uniform_int_distribution<> dis(0, N - 1);
    std::unordered_set<int> elems;

    while (elems.size() < k) {
        elems.insert(dis(gen));
    }

    return elems;
}

std::vector<int> pick(int N, int k) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::unordered_set<int> elems = pickSet(N, k, gen);

    // ok, now we have a set of k elements. but now
    // it's in a [unknown] deterministic order.
    // so we have to shuffle it:

    std::vector<int> result(elems.begin(), elems.end());
    std::shuffle(result.begin(), result.end(), gen);
    return result;
}

bool search_vector(vector<int> sample, int node) {
    if(std::find(sample.begin(), sample.end(), node) != sample.end()) {
        /* v contains x */
        return true;
    }
    else {
        return false;  
    /* v does not contain x */
    }
}

int find_ele(vector<int> sample, int ele) {

    for (int i = 0; i < sample.size(); ++i) {
        if (sample.at(i) == ele) {
            return i;
        }
    }

    return -1;
}

void sample_nodes(int sample_size, char in [100], char out [100]) {

    int n;
    int num_edges;

    ifstream og(in);
    og >> n;
    og >> num_edges;

    if (n < sample_size){
        cout << "Sampling failure: size of graph smaller than sampling size.\n";
        cout.flush();
        exit(EXIT_FAILURE);
    }

    vector<int> sample = pick(n, sample_size);
    sort(sample.begin(), sample.end());

    vector <tuple<int, int>> edges;
    int edge_1;
    int edge_2;
    tuple<int, int> tup;

    int node1;
    int node2;
    int count = 0;

    while (og >> edge_1 >> edge_2) {
        node1 = find_ele(sample, edge_1);
        node2 = find_ele(sample, edge_2);
        if ((node1 != -1) & (node2 != -1)) {
            tup = make_tuple(node1, node2);
            edges.push_back(tup);
            ++count;
            }
    }
    og.close();

    ofstream out_file(out);

    out_file << sample_size << '\n';
    out_file << count;

    for (auto it = edges.begin(); it != edges.end(); ++it) {
        out_file << '\n' << get<0>(*it) << ' ' << get<1>(*it);
    }
    out_file.close();

}

void sample_edges(int sample_edges, char in [100], char out [100]) {

    int n;
    int num_edges;

    ifstream og(in);
    og >> n;
    og >> num_edges;

    if (num_edges < sample_edges){
        cout << "Sampling failure: edges of graph less than sampling edges.\n";
        cout.flush();
        exit(EXIT_FAILURE);
    }

    vector<int> sample = pick(num_edges, sample_edges);

    vector <tuple<int, int>> edges;
    int edge_1;
    int edge_2;
    tuple<int, int> tup;

    bool edge_check;
    int ind = 0;

    while (og >> edge_1 >> edge_2) {
        edge_check = search_vector(sample, ind);
        if (edge_check) {
            tup = make_tuple(edge_1, edge_2);
            edges.push_back(tup);
            }
        ++ind;
    }
    og.close();

    ofstream out_file(out);

    out_file << n << '\n';
    out_file << sample_edges;

    for (auto it = edges.begin(); it != edges.end(); ++it) {
        out_file << '\n' << get<0>(*it) << ' ' << get<1>(*it);
    }
    out_file.close();

}

void all_comp(string file_list, bool do_vert, bool do_gdd, int iterations, bool do_outerloop, bool calc_auto, bool verbose) {
  
  int mot_min = 3;
  int mot_mx = 10;
  string directory = "small_fb";

  vector<string> graphs;

  ifstream file(file_list);
  string line;
  while (getline(file, line)) {;
      graphs.push_back(line);
  }
  file.close();

  char nameA [100];
  char nameB [100];
  char graph_fileA [100];
  char graph_fileB [100];
  string temp_path;
  double comp_val;

  for (int n = mot_min; n <= mot_mx; ++n) {
    for (int j = 0; j < graphs.size(); ++j) {
      sprintf(nameB, graphs.at(j).c_str());
      sprintf (graph_fileB, "%s/%s", directory.c_str(), nameB);
      for (int i = 0; i < j + 1; ++i) {
        sprintf(nameA, graphs.at(i).c_str());
        sprintf (graph_fileA, "%s/%s", directory.c_str(), nameA);
          comp_val = run_compare_graphs(graph_fileA, graph_fileB, n, 
                  do_vert, do_gdd, 
                  iterations, do_outerloop, calc_auto, 
                  verbose, false, 0, false, true);
          cout << nameA << "," << nameB << "," << n << "," << comp_val << "\n";
      }
    }
  }

  cout.flush();

}

void samp_graphs(char* graph, int iterations, int it_edges, int samp_nodes, int samp_edges) {

  string directory_in = "small_fb";
  string directory_out = "samp_run";

  char in [100];
  char out_n [100];
  char out_e [100];

  int count = 1;
  sprintf(in, "%s/%s", directory_in.c_str(), graph);
    for (int node_rep = 1; node_rep <= iterations; ++node_rep) {
      sprintf(out_n, "%s/n%d_%s", directory_out.c_str(), node_rep, graph);
      sample_nodes(samp_nodes, in, out_n);
      for (int edge_rep = 1; edge_rep <= it_edges; ++edge_rep) {
        sprintf(out_e, "%s/e%d_%s", directory_out.c_str(), count, graph);
        sample_edges(samp_edges, out_n, out_e);
        ++count;
      }
    }
}

void samp_graphs_lst(string file_list, int iterations, int it_edges, int samp_nodes, int samp_edges) {

  ifstream file(file_list);
  string line;
  char name [100];

  while (getline(file, line)) {
    strncpy(name, line.c_str(), 100);
    samp_graphs(name, iterations, it_edges, samp_nodes, samp_edges);
  }
  file.close();
}

double samp_comp(char* graph_fileA, char* graph_fileB, int motif, 
                bool do_outerloop, int iterations, int it_edges, int samp_nodes, int samp_edges, int comp_it, bool calc_auto, bool isCentered, bool generateGraphs)
{
  
  double elt;
  if (timing) {
    elt = timer();
  }

  if (generateGraphs) {
    samp_graphs(graph_fileA, iterations, it_edges, samp_nodes, samp_edges);
    samp_graphs(graph_fileB, iterations, it_edges, samp_nodes, samp_edges);
  }

  string directory = "small_fb";
  string out_dir = "samp_run";

  char in_1 [100];
  sprintf(in_1, "%s/%s", directory.c_str(), graph_fileA);
  char in_2 [100];
  sprintf(in_2, "%s/%s", directory.c_str(), graph_fileB);

  int n_A;
  int edges_A;
  ifstream fileA(in_1);
  fileA >> n_A;
  fileA >> edges_A;
  fileA.close();

  int n_B;
  int edges_B;
  ifstream fileB(in_2);
  fileB >> n_B;
  fileB >> edges_B;
  fileB.close();

  if (min(n_A, n_B) < samp_nodes) {
    cout << "Too small to sample, at least one graph has nodes less than " << samp_nodes << ".\n";
    samp_nodes = min(n_A, n_B) / 2;
    cout << "Sampling with nodes " << samp_nodes << " instead.\n";
    cout.flush();
  }

  double simi_val;
  char out_e_1 [100];
  char out_e_2 [100];

  vector<double> simi_scores;

  for (int r1 = 1; r1 <= iterations * it_edges; ++r1) {
    for (int r2 = 1; r2 <= iterations * it_edges; ++r2) {
      sprintf(out_e_1, "%s/e%d_%s", out_dir.c_str(), r1, graph_fileA);
      sprintf(out_e_2, "%s/e%d_%s", out_dir.c_str(), r2, graph_fileB);
      simi_val = run_algorithm2(out_e_1, out_e_2, motif, 
                false, false, false,
                comp_it, do_outerloop, calc_auto, 
                false, false, 0, false, isCentered);
      simi_scores.push_back(simi_val);
    }
  }

  double avg = 0;
  cout << graph_fileA << "," << graph_fileB << "," << motif << ",";
  cout.flush();
  for (int i = 0; i <  simi_scores.size(); ++i) {
    cout << simi_scores.at(i) << ",";
    cout.flush();
    avg += simi_scores.at(i);
  }

  avg = avg / simi_scores.size();
  cout << " avg: " << avg;
  cout.flush();

  if (timing) {
    cout << ", timing: " << timer() - elt;
  }

  return avg;

}

vector<double> samp_comp_all_n(char* graph_fileA, char* graph_fileB,
                bool do_outerloop, int iterations, int it_edges, int samp_nodes, int samp_edges, int comp_it, bool calc_auto, bool isCentered, bool generateGraphs)
{

  if (generateGraphs) {
    samp_graphs(graph_fileA, iterations, it_edges, samp_nodes, samp_edges);
    samp_graphs(graph_fileB, iterations, it_edges, samp_nodes, samp_edges);
  }

  vector<double> vals;
  double temp_val;

  int min_motif = 5;
  int max_motif = 5;

  for (int motif = min_motif; motif <= max_motif; ++motif) {
    temp_val = samp_comp(graph_fileA, graph_fileB, motif, do_outerloop, iterations, it_edges, samp_nodes, samp_edges, comp_it, calc_auto, isCentered, false);
    vals.push_back(temp_val);
    cout << "\n";
    cout.flush();
  }

  return vals;
}

void samp_all_dcc(string file_lst, 
                bool do_outerloop, int iterations, int it_edges, int samp_nodes, int samp_edges, int comp_it, bool calc_auto, bool isCentered)
{

  string directory_in = "small_fb";
  ifstream file(file_lst);

  string line;
  string file1;
  string file2;
  char in1 [100];
  char in2 [100];

  while (getline(file, line)) {
    stringstream ss(line);
    getline(ss, file1, ',');
    getline(ss, file2, ',');
    strncpy(in1, file1.c_str(), 100);
    strncpy(in2, file2.c_str(), 100);
    samp_comp_all_n(in1, in2, do_outerloop, iterations, it_edges, samp_nodes, samp_edges, comp_it, calc_auto, isCentered, false);
  }
  file.close();

}

void all_trees(char* graph_file, char* out, int iterations, 
        bool do_outerloop, bool isCentered) {

  int min_motif = 5;
  int max_motif = 5;
  double elt;

  ofstream file(out);
  file << graph_file << '\n';
  vector<double> tree_counts;

  for (int i = min_motif; i <= max_motif; ++i) {
    if (timing) {
    elt = timer();
    }
    for (int rep = 0; rep < iterations; ++rep) {
      tree_counts = run_motif(graph_file, i, false, false, 1, do_outerloop, true, false, false, 0, false, isCentered, 2);
      for (int j = 0; j < tree_counts.size(); ++j) {
        file << tree_counts.at(j);
        if (j != tree_counts.size() - 1) {
          file << ",";
        }
      }
      if (rep != iterations -1) {
        file << "||";
      }
      file.flush();
      std::vector<double>().swap(tree_counts);
    }
    if (timing) {
      cout << "time, " << graph_file << ", " << i << ", " << timer() - elt << "\n";
      cout.flush();
    }
    file << "\n\n";
    file.flush();
  }

  file.close();
  
}

void trees_for_graphs(string file_list, int iterations, 
        bool do_outerloop, bool isCentered) {

  string directory_in = "small_fb";
  string directory_out = "count_trees_no";
  if (do_outerloop) {
    directory_out = "count_trees_o";
  }

  ifstream file(file_list);

  string line;
  char in [100];
  char out [100];

  while (getline(file, line)) {
    sprintf (in, "%s/%s", directory_in.c_str(), line.c_str());
    sprintf (out, "%s/counts_%s", directory_out.c_str(), line.c_str());
    all_trees(in, out, iterations, do_outerloop, isCentered);
  }
  file.close();

}

int main(int argc, char** argv)
{
  // remove buffer so all outputs show up before crash
  srand ( time(NULL) );
  setbuf(stdout, NULL);

  char* graph_fileA = NULL;
  char* graph_fileB = NULL;
  char* template_file = NULL;
  char* batch_file = NULL;
  int iterations = 1;
  bool do_outerloop = false;
  bool calculate_automorphism = true;
  bool labeled = false;
  bool do_gdd = false;
  bool do_vert = false;
  bool verbose = false;
  bool compare_graphs = false;
  bool sim_1 = false;
  bool sim2_corr = false;
  bool sim2_ind = false;
  int klow = 0;
  bool many_comp = false;
  bool small_sample = false;
  bool all_n_sample = false;
  bool samp_from_lst = false;
  bool count_trees = false;
  int it_edges = 1;
  int samp_nodes = 1000;
  int samp_edges = 1000;
  int comp_it = 1;
  int motif = 0;
  int n = 0;
  float p = 0.0;
  float s = 0.0;
  int m = 0;
  bool isCentered = false;
  bool generateGraphs = false;
  bool samp_first = false;
  bool sophie = false;

  char c;
  while ((c = getopt (argc, argv, "g:f:t:b:m:n:p:s:j:i:k:A:B:C:H:SGDEFuWXqacdvrohlxyze")) != -1)
  {
    switch (c)
    {
      case 'h':
        print_info(argv[0]);
        break;
      case 'l':
        labeled = true;
        break;
      case 'g':
        graph_fileA = strdup(optarg);
        break;
      case 'f':
        graph_fileB = strdup(optarg);
        break;
      case 't':
        template_file = strdup(optarg);
        break;
      case 'b':
        batch_file = strdup(optarg);
        break;
      case 'm':
        m = atoi(optarg);
        break;
      case 'n':
        n = atoi(optarg);
        break;
      case 'p':
        p = atof(optarg);
        break;
      case 's':
        s = atof(optarg);
        break;
      case 'i':
        iterations = atoi(optarg);
        break;
      case 'j':
        klow = atoi(optarg);
        break;
      case 'k':
        motif = atoi(optarg);
        break;
      case 'A':
        it_edges = atoi(optarg);
        break;
      case 'B':
        samp_nodes = atoi(optarg);
        break;
      case 'C':
        samp_edges = atoi(optarg);
        break;
      case 'H':
        comp_it = atoi(optarg);
        break;
      case 'S':
        sophie = true;
        break;
      case 'G':
        generateGraphs = true;
        break;
      case 'D':
        isCentered = true;
        break;
      case 'u':
        sim_1 = true;
        break;
      case 'W':
        sim2_corr = true;
        break;
      case 'X':
        sim2_ind = true;
        break;
      case 'q':
        compare_graphs = true;
        break;
      case 'x':
        many_comp = true;
        break;
      case 'y':
        small_sample = true;
        break;
      case 'e':
        all_n_sample = true;
        break;
      case 'F':
        samp_from_lst = true;
        break;
      case 'E':
        samp_first = true;
        break;
      case 'z':
        count_trees = true;
        break;
      case 'a':
        calculate_automorphism = false; 
        break;
      case 'c':
        do_vert = true;
        break;
      case 'd':
        do_gdd = true;
        break;
      case 'o':
        do_outerloop = true;
        break;
      case 'v':
        verbose = true;
        break;
      case 'r':
        timing = true;
        break;
      case '?':
        if (optopt == 'g' || optopt == 't' || optopt == 'b' || optopt == 'i' || optopt == 'm')
          fprintf (stderr, "Option -%c requires an argument.\n", optopt);
        else if (isprint (optopt))
          fprintf (stderr, "Unknown option `-%c'.\n", optopt);
        else
          fprintf (stderr, "Unknown option character `\\x%x'.\n",
      optopt);
        print_info(argv[0]);
      default:
        abort();
    }
  } 
  if(!generateGraphs && !sim_1 && !sim2_corr && !sim2_ind && !many_comp && !small_sample && !count_trees && !all_n_sample && !samp_first && !samp_from_lst)
  {
    if(argc < 3)
    {
        printf("%d",sim_1);
        print_info_short(argv[0]);
    }

    if (motif && (motif < 3 || motif > 10))
    {
      printf("\nMotif option must be between [3,10]\n");    
      print_info(argv[0]);
    }
    else if (graph_fileA == NULL)
    { 
      printf("\nMust supply graph file\n");    
      print_info(argv[0]);
    }
    else if (template_file == NULL && batch_file == NULL && !motif)
    {
      printf("\nMust supply template XOR batchfile or -m option\n");
      print_info(argv[0]);
    }  
    else if (template_file != NULL && batch_file != NULL)
    {
      printf("\nMust only supply template file XOR batch file\n");
      print_info(argv[0]);
    }
    else if (iterations < 1)
    {
      printf("\nNumber of iterations must be positive\n");    
      print_info(argv[0]);
    }
  }

  if(generateGraphs && !small_sample && !all_n_sample && !samp_from_lst) {
    if(n && p && s && m) {
      printf("%d %f %f %d", n, p, s, m);
      generate_both_graphs(n, p, s, m);
    } else {
      printf("\nMissing Arguments\n");
      printf("%d %f %f %d", n, p, s, m);
    }
  }
  else if(sim_1) {
    printf("%d %d %f", m, iterations, p);
    sim1(m, iterations, p, true, isCentered);
  }
  else if(sim2_corr) {
    const char folderCorr [] = "sim2_corr/";
    char graphACorr [100];
    sprintf(graphACorr, "%s%d_%dA_%.5f_%.5f_corr.txt", folderCorr, m, n, p, s);
    char graphBCorr [100];
    sprintf(graphBCorr, "%s%d_%dB_%.5f_%.5f_corr.txt", folderCorr, m, n, p, s);
    if(n && p && s && m && iterations) {
        // printf("%d %f %f %d %d %d %d %d correlated Sophie: %d", n, p, s, klow, motif, m, iterations, isCentered, sophie);
        sim2(graphACorr, graphBCorr, n, p, s, klow, motif, m, iterations, isCentered, sophie);
    }
    else{
      printf("\nMissing Arguments\n");
      printf("%d %f %f %d %d %d %d", n, p, s, klow, motif, m, iterations);
    }
  }
  else if(sim2_ind) {
    const char folderInd [] = "sim2_ind/";
    char graphAInd [100];
    sprintf(graphAInd, "%s%d_%dA_%.5f_%.5f_ind.txt", folderInd, m, n, p, s);
    char graphBInd [100];
    sprintf(graphBInd, "%s%d_%dB_%.5f_%.5f_ind.txt", folderInd, m, n, p, s);
    if(n && p && s && m && iterations) {
        // printf("%d %f %f %d %d %d %d %d indepedent Sophie: %d", n, p, s, klow, motif, m, iterations, isCentered, sophie);
        sim2(graphAInd, graphBInd, n, p, s, klow, motif, m, iterations, isCentered, sophie);
    }
    else{
      printf("\nMissing Arguments\n");
      printf("%d %f %f %d %d %d %d", n, p, s, klow, motif, m, iterations);
    } 
  }
  else if(compare_graphs && motif && sophie) {
    if(sophie) {
        run_algorithm2(graph_fileA, graph_fileB, motif, labeled,
              do_vert, do_gdd, 
              iterations, do_outerloop, calculate_automorphism, 
              verbose, false, 0, true, isCentered);

    } else {
      run_compare_graphs(graph_fileA, graph_fileB, motif,
              do_vert, do_gdd, 
              iterations, do_outerloop, calculate_automorphism, 
              verbose, false, 0, true, isCentered);
    }

  }
  else if(many_comp) {
    all_comp(graph_fileA, do_vert, do_gdd, iterations, do_outerloop, calculate_automorphism, verbose);
  }
  else if (small_sample) {
    samp_comp(graph_fileA, graph_fileB, motif, do_outerloop, iterations, it_edges, samp_nodes, samp_edges, comp_it, calculate_automorphism, isCentered, generateGraphs);
  }
  else if (samp_first && (graph_fileB == NULL)) {
    samp_graphs_lst(graph_fileA, iterations, it_edges, samp_nodes, samp_edges);
  }
  else if (samp_first && (graph_fileB != NULL)) {
    samp_graphs(graph_fileA, iterations, it_edges, samp_nodes, samp_edges);
    samp_graphs(graph_fileB, iterations, it_edges, samp_nodes, samp_edges);
  }
  else if (all_n_sample) {
    samp_comp_all_n(graph_fileA, graph_fileB, do_outerloop, iterations, it_edges, samp_nodes, samp_edges, comp_it, calculate_automorphism, isCentered, generateGraphs);
  }
  else if (samp_from_lst) {
    samp_all_dcc(graph_fileA, do_outerloop, iterations, it_edges, samp_nodes, samp_edges, comp_it, calculate_automorphism, isCentered);
  }
  else if (count_trees) {
    trees_for_graphs(graph_fileA, iterations, do_outerloop, isCentered);
  }
  else if (motif)
  {
    run_motif(graph_fileA, motif, 
              do_vert, do_gdd, 
              iterations, do_outerloop, calculate_automorphism, 
              verbose, false, 0, true, isCentered, 2);
  }
  else if (template_file != NULL)
  {
    run_single(graph_fileA, template_file, labeled,                
                do_vert, do_gdd,
                iterations, do_outerloop, calculate_automorphism,
                verbose, false, 0, true, isCentered);
  }
  else if (batch_file != NULL)
  {
    run_batch(graph_fileA, batch_file, labeled,
                do_vert, do_gdd,
                iterations, do_outerloop, calculate_automorphism,
                verbose, false, 0, true, isCentered, 2);
  }

  return 0;
}
