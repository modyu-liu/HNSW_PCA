#include "hnsw.h"
#include "pca.h"

HNSWContext *hnsw_init_context(const char *filename, size_t dim, size_t len , int M = 6, int Mmax = 12)
{
    HNSWContext *ctx = (HNSWContext *)malloc(sizeof(HNSWContext));
    ctx->dim = dim;
    ctx->len = len;
    ctx->data = (VecData *)malloc(sizeof(VecData) * len);
    ctx->layer = (size_t *)malloc(sizeof(size_t) * len);
    // init file context
    FileContext *f_ctx = init_file_context(filename);
    //cout<<"check::" << ' '<<M<<' '<<log(M)<<'\n';

    double ml = 1 / (log(M));

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double>dis(0.0, 1.0);
    int maxlayer = 0;
    MatrixXd X(len , dim);
    for (int i = 0; i < len; i++)
    {
        ctx->data[i].id = i;
        ctx->data[i].vec = (float *)malloc(sizeof(float) * GLOBAL_DIM);
        read_vec_data(f_ctx, ctx->data[i].vec);
        for(int j = 0; j < dim ; j++){
            X(i , j) = ctx->data[i].vec[j];
        }
        double randomNum = dis(gen);
        int l = (int)(-log(randomNum) * ml);
        ctx->layer[i] = l;
        auto g = vector<vector<int>>(l + 1 , vector<int>());
        ctx->edg.emplace_back(g);


    }
    Pca pca(X);

    MatrixXd x = pca.result;
    ctx->lowdim = x.cols();


    ctx->lowdata = (VecData *)malloc(sizeof(VecData) * len);
    for(int i = 0; i < len ; i++){
        ctx->lowdata[i].id = i;
        ctx->lowdata[i].vec = (float *)malloc(sizeof(float) * ctx->lowdim);
        for(int j = 0; j < ctx->lowdim ; j++){
            ctx->lowdata[i].vec[j] = x(i , j);
        }
    }



    free_file_context(f_ctx);
    // need build graph
    int top = -1;
    for(int i = 0; i < len ; i++){
        int flag = Insert(ctx , i , top , ml , M , Mmax);
        if(flag != -1){
            top = flag;
        }
    }
    ctx->enter = top;
    return ctx;
}

int Insert(HNSWContext *ctx , int idx , int top , double ml , int M , int Mmax){
    int l = ctx->layer[idx];
    vector<int>W;
    int L ;
    if(top == -1){
        ctx->layer[idx] = l;
        return idx;
    }
    else {
        L = ctx->layer[top];
        gq ep;
        ep.push(make_pair(distance(ctx->lowdata[top] , ctx->lowdata[idx] , ctx->lowdim) , top));
        //cout<<"check::top1 "<<top<<'\n';
        for(int lc = L ; lc > l ; lc --){
            search_layer(ctx , ctx->lowdata[idx] , ep , 1 , lc);
        }
        for(int lc = min(l , L) ; lc >= 0 ; lc--){
            search_layer(ctx, ctx->lowdata[idx] , ep , 100 , lc);
            auto neighbors = select_neighbors( ctx , ep , M);
            // 检查每个neighbors的连接数，如果大于Mmax，则需要缩减连接到最近邻的Mmax个
            sort(neighbors.begin() , neighbors.end());
            for(auto e : neighbors){
                ctx->edg[idx][lc].emplace_back(e);
                ctx->edg[e][lc].emplace_back(idx);
                if(ctx->edg[e][lc].size() > Mmax){
                    gq q;
                    for(auto it : ctx->edg[e][lc]){
                        q.push(make_pair(distance(ctx->lowdata[it] , ctx->lowdata[e] , ctx->lowdim) , it));
                    }
                    vector<int> newneighbors = select_neighbors(ctx , q , Mmax );
                    sort(newneighbors.begin() , newneighbors.end());
                    sort(ctx->edg[e][lc].begin() , ctx->edg[e][lc].end());
                    //eNewConn ← SELECT_NEIGHBORS(e, eConn, Mmax, lc)
                    int idx1 = 0 , idx2 = 0;
                    while(idx1 < newneighbors.size()){
                        if(newneighbors[idx1] == ctx->edg[e][lc][idx2]){
                            idx1 ++;
                            idx2 ++;
                        }
                        else {
                            int it = ctx->edg[e][lc][idx2];
                            int pos = lower_bound(ctx->edg[it][lc].begin() , ctx->edg[it][lc].end() , e) - ctx->edg[it][lc].begin();
                            ctx->edg[it][lc].erase(ctx->edg[it][lc].begin() + pos);
                            idx2++;
                        }
                    }
                    while(idx2 < ctx->edg[e][lc].size()){
                        int it = ctx->edg[e][lc][idx2];
                        int pos = lower_bound(ctx->edg[it][lc].begin() , ctx->edg[it][lc].end() , e) - ctx->edg[it][lc].begin();
                        ctx->edg[it][lc].erase(ctx->edg[it][lc].begin() + pos);
                        idx2++;
                    }
                    ctx->edg[e][lc] = newneighbors;
                }
            }
        }
        //cout<<"finish!"<<'\n';
    }
    if(l > L)return idx;
    else return -1;

}
void search_layer(HNSWContext *ctx ,  VecData &q , gq& ep , int ef , int lc){

    set<int>v;
    gq C;
    lq Q;


    while(!ep.empty()){
        auto now = ep.top();
        ep.pop();
        C.push(now);
        Q.push(now);
        v.insert(now.second);

    }

    while(!C.empty()){
        auto c = C.top();C.pop();
        auto f = Q.top();
        double cq = distance(ctx->lowdata[c.second] , q , ctx->lowdim);
        double fq = distance(ctx->lowdata[f.second] , q , ctx->lowdim);
        if(cq > fq){
            break;
        }

        for(auto e : ctx->edg[c.second][lc]){
            if(v.find(e) == v.end()){
                v.insert(e);
                double eq = distance(ctx->lowdata[e] , q , ctx->lowdim);
                if(eq < fq || Q.size() < ef){
                    C.push(make_pair(eq , e));
                    Q.push(make_pair(eq , e));
                    if(Q.size() > ef){
                        Q.pop();
                    }
                }
            }
        }
    }
    while(!Q.empty()){
        ep.push(Q.top());
        Q.pop();
    }
}
void search_layer_knn(HNSWContext *ctx ,  VecData &q , gq& ep , int ef , int lc){
    set<int>v;
    gq C;
    lq Q;
    while(!ep.empty()){
        auto now = ep.top();
        ep.pop();
        C.push(now);
        Q.push(now);
        v.insert(now.second);

    }

    while(!C.empty()){
        auto c = C.top();C.pop();
        auto f = Q.top();
        double cq = distance(ctx->data[c.second] , q , ctx->dim);
        double fq = distance(ctx->data[f.second] , q , ctx->dim);
        if(cq > fq){
            break;
        }

        for(auto e : ctx->edg[c.second][lc]){
            if(v.find(e) == v.end()){
                v.insert(e);
                double eq = distance(ctx->data[e] , q , ctx->dim);
                if(eq < fq || Q.size() < ef){
                    C.push(make_pair(eq , e));
                    Q.push(make_pair(eq , e));
                    if(Q.size() > ef){
                        Q.pop();
                    }
                }
            }
        }
    }
    while(!Q.empty()){
        ep.push(Q.top());
        Q.pop();
    }
}
double distance(VecData &d1 , VecData &d2 , int dim){
    double res = 0;
    for(int i = 0; i < dim ; i++){
        res += (d1.vec[i] - d2.vec[i]) * (d1.vec[i] - d2.vec[i]);
    }
    return sqrt(res);
}

vector<int> select_neighbors(HNSWContext *ctx , gq C , int M ){

    vector<int>R;
    gq wR;

    while(!C.empty() && R.size() < M){
        auto e = C.top();
        C.pop();
        if(R.size() == 0){
            R.emplace_back(e.second);
            continue;
        }
        bool ok = 1 ;
        for(auto it : R){
            if(distance(ctx->lowdata[e.second] , ctx->lowdata[it] , ctx->lowdim) < e.first){
                ok = 0;
                break;
            }
        }
        if(ok){
            R.emplace_back(e.second);
        }
        else {
            wR.push(e);
        }

    }
    while(R.size() < M && !wR.empty()){
        R.emplace_back(wR.top().second);
        wR.pop();
    }

    return R;
}


void hnsw_approximate_knn(HNSWContext *ctx, VecData &q, int *results, int k)
{
    // sort existing vectors
    int enter = ctx->enter;
    int L = ctx->layer[enter];
    gq ep;
    ep.push(make_pair(distance(ctx->data[enter] , q , ctx->dim) , enter));
    for(int lc = L ; lc > 0 ; lc--){
        search_layer_knn(ctx , q , ep , 1 , lc);
    }
    search_layer_knn(ctx , q , ep , k , 0);

    int idx = 0;
    while(!ep.empty()){
        auto now = ep.top();
        ep.pop();
        results[idx++] = now.second;

    }

}
