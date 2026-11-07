#include <cublasLt.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

struct Session {
    cublasLtHandle_t handle{};
    cublasLtMatmulDesc_t op{};
    cublasLtMatrixLayout_t a{}, b{}, c{};
    cublasLtMatmulPreference_t pref{};
    std::vector<cublasLtMatmulHeuristicResult_t> results;
    size_t cap{};
};
static int layout(cublasLtMatrixLayout_t* p, int64_t rows, int64_t cols, int64_t s0, int64_t s1) {
    cublasLtOrder_t order;
    int64_t ld;
    if(s1==1) { order=CUBLASLT_ORDER_ROW; ld=s0; }
    else if(s0==1) { order=CUBLASLT_ORDER_COL; ld=s1; }
    else return -101;
    int rc=cublasLtMatrixLayoutCreate(p, CUDA_R_16BF, rows, cols, ld);
    if(rc) return rc;
    return cublasLtMatrixLayoutSetAttribute(*p,CUBLASLT_MATRIX_LAYOUT_ORDER,&order,sizeof(order));
}
extern "C" {
void* make_session(int64_t m,int64_t n,int64_t k,int64_t a0,int64_t a1,int64_t b0,int64_t b1,size_t cap,int* status) {
    Session* s=new Session();s->cap=cap;
    // Descriptor computes the same logical A@B; no tensor copy or transpose is performed.
    int rc=cublasLtCreate(&s->handle);
    if(!rc) rc=cublasLtMatmulDescCreate(&s->op,CUBLAS_COMPUTE_32F,CUDA_R_32F);
    if(!rc) rc=layout(&s->a,m,k,a0,a1);
    if(!rc) rc=layout(&s->b,k,n,b0,b1);
    if(!rc) rc=layout(&s->c,m,n,n,1);
    if(!rc) rc=cublasLtMatmulPreferenceCreate(&s->pref);
    if(!rc) rc=cublasLtMatmulPreferenceSetAttribute(s->pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&cap,sizeof(cap));
    s->results.resize(32);int count=0;
    if(!rc) rc=cublasLtMatmulAlgoGetHeuristic(s->handle,s->op,s->a,s->b,s->c,s->c,s->pref,32,s->results.data(),&count);
    s->results.resize(count);*status=rc;return s;
}
int count_algos(void* ptr) { return static_cast<Session*>(ptr)->results.size(); }
int describe_algo(void* ptr,int index,int* id,int* split,int* reduction,size_t* workspace,float* waves) {
    auto& h=static_cast<Session*>(ptr)->results.at(index);size_t written=0;
    *id=*split=*reduction=-1;*workspace=h.workspaceSize;*waves=h.wavesCount;
    int rc=cublasLtMatmulAlgoConfigGetAttribute(&h.algo,CUBLASLT_ALGO_CONFIG_ID,id,sizeof(*id),&written);
    if(!rc) rc=cublasLtMatmulAlgoConfigGetAttribute(&h.algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,split,sizeof(*split),&written);
    if(!rc) rc=cublasLtMatmulAlgoConfigGetAttribute(&h.algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,reduction,sizeof(*reduction),&written);
    return rc ? rc : h.state;
}
int run_algo(void* ptr,int index,void* a,void* b,void* out,void* workspace,void* stream) {
    auto* s=static_cast<Session*>(ptr);float alpha=1.0f,beta=0.0f;
    return cublasLtMatmul(s->handle,s->op,&alpha,a,s->a,b,s->b,&beta,out,s->c,out,s->c,
                          &s->results.at(index).algo,workspace,s->results.at(index).workspaceSize,static_cast<cudaStream_t>(stream));
}
size_t algo_bytes() { return sizeof(cublasLtMatmulAlgo_t); }
int copy_algo(void* ptr,int index,void* bytes,size_t size) {
    if(size!=sizeof(cublasLtMatmulAlgo_t)) return -102;
    std::memcpy(bytes,&static_cast<Session*>(ptr)->results.at(index).algo,size);return 0;
}
int install_algo(void* ptr,const void* bytes,size_t size) {
    if(size!=sizeof(cublasLtMatmulAlgo_t)) return -102;
    auto* s=static_cast<Session*>(ptr);cublasLtMatmulAlgo_t algo;
    std::memcpy(&algo,bytes,size);cublasLtMatmulHeuristicResult_t checked{};
    int rc=cublasLtMatmulAlgoCheck(s->handle,s->op,s->a,s->b,s->c,s->c,&algo,&checked);
    if(rc || checked.state || checked.workspaceSize>s->cap) return rc ? rc : -103;
    checked.algo=algo;s->results={checked};return 0;
}
void drop_session(void* ptr) {
    auto* s=static_cast<Session*>(ptr);
    if(s->pref) cublasLtMatmulPreferenceDestroy(s->pref);
    if(s->a) cublasLtMatrixLayoutDestroy(s->a);
    if(s->b) cublasLtMatrixLayoutDestroy(s->b);
    if(s->c) cublasLtMatrixLayoutDestroy(s->c);
    if(s->op) cublasLtMatmulDescDestroy(s->op);
    if(s->handle) cublasLtDestroy(s->handle);
    delete s;
}
}
