#pragma once

#include "utils.h"

// you can add more structures here or modify existing structrues.

typedef struct HNSWGraph
{
    size_t layers_num; // number of layers
} HNSWGraph;

typedef struct HNSWContext
{
    size_t enter;
    size_t dim;       // dimension of dataset
    size_t lowdim;

    size_t len;       // size of dataset
    VecData *data;    // vectors will be loaded into this array
    VecData *lowdata;

    size_t *layer; // graph of HNSW

    vector<vector<vector<int>>> edg;


} HNSWContext;
typedef priority_queue<pair<double , int> , vector<pair<double , int>> , less<pair<double , int>>> lq;
typedef priority_queue<pair<double , int> , vector<pair<double , int>> , greater<pair<double , int>>> gq;


// you can declare some help functions here, and implement them in 'hnsw.c'

// public functions here
// Please do not modify these function signatures!
// To simply our program, we do not consider reclaiming memory space here.
// TODO: Please implement these functions according to HNSW algorithm.
HNSWContext *hnsw_init_context(const char *filename, size_t dim, size_t len , int M , int Mmax);
void hnsw_approximate_knn(HNSWContext *ctx, VecData &q, int *results, int k);
int Insert(HNSWContext *ctx , int idx , int top , double ml , int M , int Mmax);
vector<int> select_neighbors(HNSWContext *ctx , gq C , int M );
double distance(VecData &d1 , VecData &d2 , int dim);
void search_layer(HNSWContext *ctx ,  VecData &q , gq& ep , int ef , int lc);
void search_layer_knn(HNSWContext *ctx ,  VecData &q , gq& ep , int ef , int lc);