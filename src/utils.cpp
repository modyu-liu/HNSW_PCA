#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

float vec_dist(VecData x, VecData y)
{
    float sum = 0.0;
    for (size_t i = 0; i < GLOBAL_DIM; i++)
    {
        float diff = x.vec[i] - y.vec[i];
        sum += diff * diff;
    }
    return sum;
}

FileContext *init_file_context(const char *filename)
{
    FileContext *ctx = (FileContext *)malloc(sizeof(FileContext));
    ctx->filename = (char *)malloc(strlen(filename) + 1L);
    memcpy(ctx->filename, filename, strlen(filename) + 1L);

    ctx->stream = fopen(filename, "rb");
    if (ctx->stream == NULL)
    {
        fprintf(stderr, "I/O error : Unable to open the file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    ctx->offset = 0;
    return ctx;
}

void read_4bytes(FileContext *ctx, void *dst)
{
    size_t s = fread(dst, 4L, 1, ctx->stream);
    assert(s == 1L);
}

void read_vec_data(FileContext *ctx, void *dst)
{
    read_4bytes(ctx, dst);
    size_t s = fread(dst, 4L, GLOBAL_DIM, ctx->stream);
    assert(s == GLOBAL_DIM);
}

void read_id_data(FileContext *ctx, void *dst, size_t n)
{
    read_4bytes(ctx, dst);
    size_t s = fread(dst, 4L, n, ctx->stream);
    assert(s == n);
}

void free_file_context(FileContext* ctx)
{
    fclose(ctx->stream);
    free(ctx->filename);
    free(ctx);
}
