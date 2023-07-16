#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "hnsw.h"
#include "utils.h"

#define K 100

float cal_recall_value(int *results, int *trueset, int k)
{
    int cnt = 0;
    for (int i = 0; i < k; i++)
    {
        int val = results[i];
        for (int j = 0; j < k; j++)
        {
            if (val == trueset[j])
            {
                cnt++;
                break;
            }
        }
    }

    return ((float)cnt) / ((float)k);
}

int main(int argc, char *argv[])
{
    if (argc != 6)
    {
        printf("Usage: ./hnsw_test base_file_path data_size query_file_path query_size groundtruth_file_path\n");
        exit(1);
    }

    int data_size = atoi(argv[2]);
    int query_size = atoi(argv[4]);

    printf("data size: %d\nquery size: %d\n", data_size, query_size);
    // init query and groundtruth files
    FileContext *query_file_ctx = init_file_context(argv[3]);
    FileContext *gt_file_ctx = init_file_context(argv[5]);

    clock_t start, end;
    start = clock();
    int M = 16 , Mmax = 32;
    HNSWContext *ctx = hnsw_init_context(argv[1], GLOBAL_DIM, data_size , M , Mmax);
    end = clock();
    printf("HNSW Context Initialied OK!\n");
    printf("HNSW initialization cost: %.4f seconds\n", ((float)(end - start)) / CLOCKS_PER_SEC);
    VecData q_vec;
    q_vec.vec = (float *)malloc(sizeof(float) * GLOBAL_DIM);
    int q_results[K];
    int true_results[K];
    float total_recall_values = 0.0;
    printf("Benchmark started......\n");

    float query_cost = 0.0;
    cout<<"check::"<<K<<'\n';

    for (int i = 0; i < query_size; i++)
    {
        read_vec_data(query_file_ctx, q_vec.vec);
        read_id_data(gt_file_ctx, true_results, K);
        start = clock();
        hnsw_approximate_knn(ctx, q_vec, q_results, K);
        end = clock();
        query_cost += ((float)(end - start)) / CLOCKS_PER_SEC;
        total_recall_values += cal_recall_value(q_results, true_results, K);
    }

    // report query time cost
    printf("%d queries cost: %.4f seconds\n", query_size, query_cost);
    // report recall value
    printf("Recall value: %.4f\n", total_recall_values / ((float)query_size));

    free_file_context(query_file_ctx);
    free_file_context(gt_file_ctx);
    return 0;
}
