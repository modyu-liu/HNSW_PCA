#pragma once

#include<bits/stdc++.h>
#include<random>
using namespace std;


#define GLOBAL_DIM 128

typedef struct
{
    int id;
    float *vec;
} VecData;

typedef struct
{
    FILE *stream;
    char *filename;
    int offset;
} FileContext;

float vec_dist(VecData x, VecData y);
FileContext *init_file_context(const char *filename);
void read_4bytes(FileContext *ctx, void *dst);
void read_vec_data(FileContext *ctx, void *dst);
void read_id_data(FileContext *ctx, void *dst, size_t n);
void free_file_context(FileContext *ctx);
