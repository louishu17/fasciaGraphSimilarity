using namespace std;

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <random>
#include <string>
#include <fstream>
#include <tuple>

void generate_graph(int n, float p, char filename[50])
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

void exec(const char* command) {
    //runs command and prints output: found on internet

    FILE *fpipe;
    char c = 0;

    if (0 == (fpipe = (FILE*)popen(command, "r")))
    {
        perror("popen() failed.");
    }

    while (fread(&c, sizeof c, 1, fpipe))
    {
        cout << c;
    }

    pclose(fpipe);

}

void sim1() {
    //runs simulation 1

    system("make");

    float p = 0.8;
    char tree_file[] = "template.graph";
    int iterations = 100;

    cout << "\n[";

    for (int i = 0; i < 20; i++) {

        int n = (i + 1) * 50;
        char graph_file [50];
        sprintf (graph_file, "graphs/%d.txt", n);
        generate_graph(n, p, graph_file);

        char command [200];
        sprintf(command, "./fascia -g %s -t %s -i %d", graph_file, tree_file, iterations);
        exec(command);

        cout << ", ";
        cout.flush();        
        
    }

    cout<<'\b';
    cout<<'\b';
    cout<<"]\n";
    cout.flush();

}

void select_edges(float s, int m_rep, char in [50], char out [50]) {

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

void sim2_corr(int n, float p, float s, int tree_len, int m_rep) {


    const char folder [] = "sim2_corr/";
    char in [50];
    sprintf(in, "%s%d_og.txt", folder, m_rep);
    char out1 [50];
    sprintf(out1, "%s%d_%d_corr.txt", folder, m_rep, 1);
    char out2 [50];
    sprintf(out2, "%s%d_%d_corr.txt", folder, m_rep, 2);
    
    generate_graph(n, p, in);
    select_edges(s, m_rep, in, out1);
    select_edges(s, m_rep, in, out2);

    char command [200];
    sprintf(command, "./fascia -g %s -f %s -m %d -q", out1, out2, tree_len);
    exec(command);
        
}

void sim2_ind(int n, float p, float s, int tree_len, int m_rep) {

    const char folder [] = "sim2_ind/";
    char out1 [50];
    sprintf(out1, "%s%d_%d_ind.txt", folder, m_rep, 1);
    char out2 [50];
    sprintf(out2, "%s%d_%d_ind.txt", folder, m_rep, 2);
    
    generate_graph(n, p, out1);
    generate_graph(n, p, out2);

    char command [200];
    sprintf(command, "./fascia -g %s -f %s -m %d -q", out1, out2, tree_len);
    exec(command);

}

void sim2() {

    int n = 1000;
    float p = 0.1;
    float s = 1;
    int tree_len = 4;
    int m = 20;

    cout << "\n[";

    for (int m_rep = 1; m_rep < m + 1; ++m_rep) {
        sim2_corr(n, p, s, tree_len, m_rep);

        cout << ", ";
        cout.flush();
    }

    cout<<'\b';
    cout<<'\b';
    cout<<"]\n";

    cout << "\n[";

    for (int m_rep = 1; m_rep < m + 1; ++m_rep) {
        sim2_ind(n, p, s, tree_len, m_rep);

        cout << ", ";
        cout.flush();
    }

    cout<<'\b';
    cout<<'\b';
    cout<<"]\n";
    cout.flush();

}

int main(int argc, char** argv)
{

    // srand(time(NULL));
    sim1();
    //sim2();

}