// ============================================================================
// onednn.cpp — oneDNN CPU 原语委托实现
//
// 通用骨架: 连续 F32 张量 → 显式步长 md → primitive_desc(命中 oneDNN
// 内建原语缓存, JIT 只发生一次) → memory 裸指针包裹(零拷贝) → execute+wait。
// 任一步失败即 return false, 调用方回落 Eigen/ref(全算子覆盖铁律)。
// ============================================================================
#include "backends/onednn/onednn.hpp"
#include "dynlib.hpp"
#include <cstring>
#include <list>
#include <memory>
#include <vector>

namespace znn {
namespace onednn {

namespace dn = ::znn::dynlib::onednn;   // 动态加载的 dnnl_* 指针

bool available() {
    return dn::available_onednn();
}

namespace {

constexpr int kMaxDim = 8;

// 引擎/流用 dynlib 的一次性缓存句柄(免每次调用重建 —— 小算子开销大头)
struct Eng {
    dnnl_engine_t p = dn::cached_engine;
    bool ok() const { return p != nullptr; }
};

struct Stm {
    dnnl_stream_t p = dn::cached_stream;
    bool wait() { return dn::stream_wait(p) == dnnl_success; }
    bool ok() const { return p != nullptr; }
};

struct Md {
    dnnl_memory_desc_t p = nullptr;
    ~Md() { if (p) dn::md_destroy(p); }
    // 连续 F32 张量 → 显式步长 md
    bool from(const Tensor& t) {
        if (!t.is_contiguous() || t.dtype() != DType::F32
            || t.dim_count() > kMaxDim || t.dim_count() == 0) {
            return false;
        }
        dnnl_dim_t dims[kMaxDim], strides[kMaxDim];
        for (int d = 0; d < (int)t.dim_count(); ++d) {
            dims[d] = (dnnl_dim_t)t.size(d);
            strides[d] = (dnnl_dim_t)t.strides()[d];
        }
        return dn::md_create(&p, (int)t.dim_count(), dims, dnnl_f32, strides)
            == dnnl_success;
    }
};

struct Pd {
    dnnl_primitive_desc_t p = nullptr;
    ~Pd() { if (p) dn::pd_destroy(p); }
};

struct Prim {
    dnnl_primitive_t p = nullptr;
    ~Prim() { if (p) dn::primitive_destroy(p); }
};

struct Mem {
    dnnl_memory_t p = nullptr;
    ~Mem() { if (p && dn::memory_destroy_ext) dn::memory_destroy_ext(p); }
    bool wrap(const_dnnl_memory_desc_t md, dnnl_engine_t e, void* data) {
        return dn::memory_create(&p, md, e, data) == dnnl_success;
    }
};

// "双入单出(+可选 scratchpad)" 执行器; arg1 指第二输入的参数号
// (逐元素=SRC_1, 卷积/池化=WEIGHTS —— 传错则 query_md 返回空, 静默回落)
bool exec(dnnl_engine_t eng, dnnl_stream_t stm, dnnl_primitive_desc_t pd,
          void* a, void* b, void* y, int arg1 = DNNL_ARG_SRC_1) {
    Prim prim;
    if (dn::primitive_create(&prim.p, pd) != dnnl_success) {
        return false;
    }
    const bool binary_op = b != nullptr;
    if (binary_op
        && dn::pd_query_md(pd, dnnl_query_exec_arg_md, arg1) == nullptr) {
        return false;   // 描述符与参数不符(防御)
    }
    const_dnnl_memory_desc_t sp_md =
        dn::pd_query_md(pd, dnnl_query_scratchpad_md, 0);
    Mem ma, mb, my, msp;
    if (!ma.wrap(dn::pd_query_md(pd, dnnl_query_exec_arg_md, DNNL_ARG_SRC_0), eng, a)
        || (binary_op
            && !mb.wrap(dn::pd_query_md(pd, dnnl_query_exec_arg_md, arg1), eng, b))
        || !my.wrap(dn::pd_query_md(pd, dnnl_query_exec_arg_md, DNNL_ARG_DST), eng, y)
        || (sp_md && !msp.wrap(sp_md, eng, DNNL_MEMORY_ALLOCATE))) {
        return false;   // RAII 全量析构
    }
    dnnl_exec_arg_t args[4];
    int n = 0;
    args[n++] = {DNNL_ARG_SRC_0, ma.p};
    if (binary_op) args[n++] = {arg1, mb.p};
    args[n++] = {DNNL_ARG_DST, my.p};
    if (sp_md) args[n++] = {DNNL_ARG_SCRATCHPAD, msp.p};
    return dn::execute(prim.p, stm, n, args) == dnnl_success
           && dn::stream_wait(stm) == dnnl_success;
}

} // namespace

// ---- 逐元素(9 算子共用; alg 为 dnnl_eltwise_*) ----
bool eltwise(Span<const Tensor*> in, Span<Tensor*> out, int alg,
             float alpha, float beta) {
    if (dn::family_off("eltwise") || !dn::available_onednn()) return false;
    Eng eng;
    Stm stm;
    if (!eng.ok() || !stm.ok()) return false;
    Md src, dst;
    if (!src.from(*in[0]) || !dst.from(*out[0])) return false;
    Pd pd;
    if (dn::eltwise_pd(&pd.p, eng.p, dnnl_forward_inference,
                       (dnnl_alg_kind_t)alg, src.p, dst.p, alpha, beta, nullptr)
        != dnnl_success) {
        return false;
    }
    return exec(eng.p, stm.p, pd.p, in[0]->data<float>(), nullptr,
                out[0]->data<float>());
}

// ---- 二元(4 算子; PyTorch 广播: 右对齐, 1 维步长 0) ----
bool binary(Span<const Tensor*> in, Span<Tensor*> out, int alg) {
    if (dn::family_off("binary") || !dn::available_onednn()) return false;
    const Tensor& a = *in[0];
    const Tensor& b = *in[1];
    const Tensor& y = *out[0];
    if (!a.is_contiguous() || !b.is_contiguous() || !y.is_contiguous()
        || a.dim_count() == 0 || b.dim_count() == 0
        || y.dim_count() > kMaxDim) {
        return false;
    }
    Eng eng;
    Stm stm;
    if (!eng.ok() || !stm.ok()) return false;
    Md ma, mb, my;
    if (!ma.from(a) || !my.from(y)) return false;
    // b 右对齐展开到 y 的秩: 1 长维步长 0(不可广播返回 false)
    dnnl_dim_t bd[kMaxDim], bs[kMaxDim];
    dnnl_dim_t acc = 1;
    const int rank = (int)y.dim_count();
    for (int d = rank - 1; d >= 0; --d) {
        const int64_t bdpos = d - (rank - (int64_t)b.dim_count());
        const int64_t bl = bdpos < 0 ? 1 : b.size(bdpos);
        if (bl != y.size(d) && bl != 1) return false;
        // Keep singleton dimensions: expanding them with zero strides forces
        // oneDNN's slow reference binary implementation on NCHW channel bias.
        bd[d] = (dnnl_dim_t)bl;
        bs[d] = acc;
        acc *= bl;
    }
    if (dn::md_create(&mb.p, rank, bd, dnnl_f32, bs) != dnnl_success) {
        return false;
    }
    Pd pd;
    if (dn::binary_pd(&pd.p, eng.p, (dnnl_alg_kind_t)alg, ma.p, mb.p, my.p,
                      nullptr) != dnnl_success) {
        return false;
    }
    return exec(eng.p, stm.p, pd.p, a.data<float>(), b.data<float>(),
                y.data<float>());
}

// ---- softmax(末维) ----
bool softmax(Span<const Tensor*> in, Span<Tensor*> out, int64_t dim) {
    if (dn::family_off("softmax") || !dn::available_onednn()) return false;
    if (!in[0]->is_contiguous() || !out[0]->is_contiguous()) return false;
    Eng eng;
    Stm stm;
    if (!eng.ok() || !stm.ok()) return false;
    Md src, dst;
    if (!src.from(*in[0]) || !dst.from(*out[0])) return false;
    Pd pd;
    if (dim < 0) dim += in[0]->dim_count();
    if (dn::softmax_pd(&pd.p, eng.p, dnnl_forward_inference,
                       (dnnl_alg_kind_t)dnnl_softmax_accurate, src.p, dst.p,
                       (int)dim, nullptr) != dnnl_success) {
        return false;
    }
    return exec(eng.p, stm.p, pd.p, in[0]->data<float>(), nullptr,
                out[0]->data<float>());
}

// ---- 规约(sum/mean 沿 dim; keepdim 由 out 形状体现) ----
bool reduce(Span<const Tensor*> in, Span<Tensor*> out, int64_t dim, bool mean) {
    if (dn::family_off("reduce") || !dn::available_onednn()) return false;
    const Tensor& x = *in[0];
    const Tensor& y = *out[0];
    if (dim < 0) dim += x.dim_count();
    if (!x.is_contiguous() || !y.is_contiguous()) return false;
    if (x.numel() < (int64_t)1 << 20) return false;   // 小规约回落(ref 更快)
    Eng eng;
    Stm stm;
    if (!eng.ok() || !stm.ok()) return false;
    Md src, dst;
    if (!src.from(x) || !dst.from(y)) return false;
    Pd pd;
    if (dn::reduction_pd(&pd.p, eng.p, (dnnl_alg_kind_t)dnnl_reduction_sum,
                         src.p, dst.p, 0.f, 0.f, nullptr) != dnnl_success) {
        return false;
    }
    if (!exec(eng.p, stm.p, pd.p, x.data<float>(), nullptr, y.data<float>())) {
        return false;
    }
    if (mean) {   // mean = sum × 1/n(复用 Eigen 快速路径元素乘)
        const int64_t n = x.size(dim);
        Tensor inv = Tensor::full(Shape{}, 1.0 / (double)n, DType::F32,
                                  Device::CPU);
        Tensor scaled = Tensor::mul(y, inv);
        std::memcpy(y.data<float>(), scaled.data<float>(),
                    (size_t)y.numel() * sizeof(float));
    }
    return true;
}

// ---- GEMM: 行主序 C = A × op(B), 列主序 dnnl_sgemm 映射(同 cuBLAS) ----
bool gemm(const float* A, int lda, const float* B, int ldb,
          float* C, int ldc, int M, int N, int K, bool trans_b) {
    if (dn::family_off("gemm") || !dn::available_onednn()) return false;
    const float alpha = 1.f, beta = 0.f;
    // dnnl_sgemm 本身就是行主序约定(文档明示, 不同于 cuBLAS 列主序):
    // C[M,N] = op(A)[M,K] × op(B)[K,N], lda/ldb/ldc 均为行距。
    // trans_b=true: B 按 [N,K] 行主序传入(权重布局), 经 'T' 用作 [K,N]
    (void)alpha; (void)beta;
    return dn::sgemm('N', trans_b ? 'T' : 'N', M, N, K, 1.f, A, lda, B, ldb,
                     0.f, C, ldc) == dnnl_success;
}

namespace {
// Plans cache only layouts, primitives and bounded temporary memory. Current
// weights are always reordered, so data() writes and optimizer updates are safe.
struct ConvPlan {
    Pd pd;
    Prim primitive,rx,rw,ry;
    Mem xe,we,ye;
    bool scratch=false;
    size_t bytes=0;
    bool init(const Tensor& x,const Tensor& w,const Tensor& y,const Conv2dOptions& o,
              const_dnnl_memory_desc_t xs,const_dnnl_memory_desc_t ws,const_dnnl_memory_desc_t ys) {
        Eng eng;Md xa,wa,ya;
        dnnl_dim_t xd[4],yd[4],wd[5];for(int i=0;i<4;++i){xd[i]=x.size(i);yd[i]=y.size(i);}
        int rank=4;
        if(o.groups>1){rank=5;wd[0]=o.groups;wd[1]=w.size(0)/o.groups;for(int i=1;i<4;++i)wd[i+1]=w.size(i);}
        else for(int i=0;i<4;++i)wd[i]=w.size(i);
        if(dn::md_tag(&xa.p,4,xd,dnnl_f32,dnnl_format_tag_any)!=dnnl_success
           ||dn::md_tag(&wa.p,rank,wd,dnnl_f32,dnnl_format_tag_any)!=dnnl_success
           ||dn::md_tag(&ya.p,4,yd,dnnl_f32,dnnl_format_tag_any)!=dnnl_success)return false;
        dnnl_dim_t st[2]={o.stride[0],o.stride[1]},di[2]={o.dilation[0]-1,o.dilation[1]-1},pad[2]={o.pad[0],o.pad[1]};
        if(dn::conv_pd(&pd.p,eng.p,dnnl_forward_inference,dnnl_convolution_direct,xa.p,wa.p,nullptr,ya.p,st,di,pad,pad,nullptr)!=dnnl_success)return false;
        auto query=[&](int arg){return dn::pd_query_md(pd.p,dnnl_query_exec_arg_md,arg);};
        auto ix=query(DNNL_ARG_SRC),iw=query(DNNL_ARG_WEIGHTS),iy=query(DNNL_ARG_DST);
        auto sd=dn::pd_query_md(pd.p,dnnl_query_scratchpad_md,0);scratch=sd!=nullptr;
        bytes=dn::md_size(ix)+dn::md_size(iw)+dn::md_size(iy)+(sd?dn::md_size(sd):0);
        if(!xe.wrap(xs,eng.p,x.data<float>())||!we.wrap(ws,eng.p,w.data<float>())||!ye.wrap(ys,eng.p,y.data<float>()))return false;
        auto reorder=[&](const_dnnl_memory_desc_t from,const_dnnl_memory_desc_t to,Prim& r){
            Pd rp;return dn::reorder_pd(&rp.p,from,eng.p,to,eng.p,nullptr)==dnnl_success
                        &&dn::primitive_create(&r.p,rp.p)==dnnl_success;
        };
        return reorder(xs,ix,rx)&&reorder(ws,iw,rw)&&reorder(iy,ys,ry)
            &&dn::primitive_create(&primitive.p,pd.p)==dnnl_success;
    }
    bool run(const Tensor& x,const Tensor& w,Tensor& y) {
        Stm stm;
        // Retain execution objects, not every layer's full activation buffers.
        // The latter increased process memory by hundreds of MB for ~1% gain.
        Eng eng;Mem xm,wm,ym,sp;
        auto query=[&](int arg){return dn::pd_query_md(pd.p,dnnl_query_exec_arg_md,arg);};
        auto sd=dn::pd_query_md(pd.p,dnnl_query_scratchpad_md,0);
        if(!xm.wrap(query(DNNL_ARG_SRC),eng.p,DNNL_MEMORY_ALLOCATE)
           ||!wm.wrap(query(DNNL_ARG_WEIGHTS),eng.p,DNNL_MEMORY_ALLOCATE)
           ||!ym.wrap(query(DNNL_ARG_DST),eng.p,DNNL_MEMORY_ALLOCATE)
           ||(sd&&!sp.wrap(sd,eng.p,DNNL_MEMORY_ALLOCATE)))return false;
        if(dn::memory_set_handle(xe.p,x.data<float>())!=dnnl_success
           ||dn::memory_set_handle(we.p,w.data<float>())!=dnnl_success
           ||dn::memory_set_handle(ye.p,y.data<float>())!=dnnl_success)return false;
        auto reorder=[&](Prim& p,Mem& from,Mem& to){
            dnnl_exec_arg_t args[2]={{DNNL_ARG_FROM,from.p},{DNNL_ARG_TO,to.p}};
            return dn::execute(p.p,stm.p,2,args)==dnnl_success && dn::stream_wait(stm.p)==dnnl_success;
        };
        if(!reorder(rx,xe,xm)||!reorder(rw,we,wm))return false;
        dnnl_exec_arg_t args[4]={{DNNL_ARG_SRC,xm.p},{DNNL_ARG_WEIGHTS,wm.p},{DNNL_ARG_DST,ym.p},{DNNL_ARG_SCRATCHPAD,sp.p}};
        if(dn::execute(primitive.p,stm.p,scratch?4:3,args)!=dnnl_success||dn::stream_wait(stm.p)!=dnnl_success)return false;
        return reorder(ry,ym,ye);
    }
};
bool conv_layout(const Tensor& x,const Tensor& w,Tensor& y,const Conv2dOptions& o,
                 const_dnnl_memory_desc_t xs,const_dnnl_memory_desc_t ws,const_dnnl_memory_desc_t ys) {
    if(!dn::md_tag || !dn::reorder_pd || !dn::md_size || !dn::memory_set_handle || dn::family_off("conv_layout"))return false;
    struct Entry {std::vector<int64_t> key;std::unique_ptr<ConvPlan> plan;};
    struct Cache {std::list<Entry> lru;size_t bytes=0;};
    thread_local Cache cache;
    std::vector<int64_t> key;
    for(int i=0;i<4;++i){key.push_back(x.size(i));key.push_back(w.size(i));key.push_back(y.size(i));}
    for(int i=0;i<2;++i){key.push_back(o.stride[i]);key.push_back(o.pad[i]);key.push_back(o.dilation[i]);}key.push_back(o.groups);
    if(!dn::family_off("conv_cache"))for(auto it=cache.lru.begin();it!=cache.lru.end();++it)if(it->key==key){
        cache.lru.splice(cache.lru.begin(),cache.lru,it);return cache.lru.front().plan->run(x,w,y);
    }
    auto plan=std::make_unique<ConvPlan>();if(!plan->init(x,w,y,o,xs,ws,ys))return false;
    constexpr size_t budget=size_t(256)*1024*1024;
    if(dn::family_off("conv_cache") || plan->bytes>budget)return plan->run(x,w,y);
    while(!cache.lru.empty() && (cache.bytes+plan->bytes>budget || cache.lru.size()>=128)){
        cache.bytes-=cache.lru.back().plan->bytes;cache.lru.pop_back();
    }
    cache.bytes+=plan->bytes;cache.lru.push_front({std::move(key),std::move(plan)});
    return cache.lru.front().plan->run(x,w,y);
}
}

// ---- 卷积: 外部 NCHW/OIHW，内部布局由 oneDNN 选择 ----
bool conv2d(const Tensor& x, const Tensor& w, Tensor& y,
            const Conv2dOptions& o) {
    if (dn::family_off("conv") || !dn::available_onednn()) return false;
    if (x.dim_count() != 4 || w.dim_count() != 4
        || o.dilation[0] != 1 || o.dilation[1] != 1) {
        return false;
    }
    Eng eng;
    Stm stm;
    if (!eng.ok() || !stm.ok()) return false;
    Md ms, mw, md;
    if (!ms.from(x) || !md.from(y)) return false;
    if (o.groups > 1) {
        if (!w.is_contiguous()) return false;
        const dnnl_dim_t dims[5] = {o.groups, w.size(0)/o.groups, w.size(1), w.size(2), w.size(3)};
        dnnl_dim_t strides[5]; strides[4] = 1;
        for (int i=3;i>=0;--i) strides[i]=strides[i+1]*dims[i+1];
        if (dn::md_create(&mw.p,5,dims,dnnl_f32,strides)!=dnnl_success) return false;
    } else if (!mw.from(w)) return false;
    if(conv_layout(x,w,y,o,ms.p,mw.p,md.p))return true;
    const dnnl_dim_t st[2] = {(dnnl_dim_t)o.stride[0], (dnnl_dim_t)o.stride[1]};
    const dnnl_dim_t di[2] = {0, 0};
    const dnnl_dim_t pl[2] = {(dnnl_dim_t)o.pad[0], (dnnl_dim_t)o.pad[1]};
    const dnnl_dim_t pr[2] = {(dnnl_dim_t)o.pad[0], (dnnl_dim_t)o.pad[1]};
    Pd pd;
    if (dn::conv_pd(&pd.p, eng.p, dnnl_forward_inference,
                    (dnnl_alg_kind_t)dnnl_convolution_direct, ms.p, mw.p,
                    nullptr, md.p, st, di, pl, pr, nullptr) != dnnl_success) {
        return false;
    }
    return exec(eng.p, stm.p, pd.p, x.data<float>(), w.data<float>(),
                y.data<float>(), DNNL_ARG_WEIGHTS);
}

// ---- 池化(空间 2D; avg 除以有效元素数) ----
bool pool2d(const Tensor& x, Tensor& y, Shape kernel, const Pool2dOptions& o,
            bool max_pool) {
    if (dn::family_off("pool") || !dn::available_onednn()) return false;
    if (x.dim_count() != 4) return false;
    Eng eng;
    Stm stm;
    if (!eng.ok() || !stm.ok()) return false;
    Md ms, md;
    if (!ms.from(x) || !md.from(y)) return false;
    const dnnl_dim_t st[2] = {(dnnl_dim_t)o.stride[0], (dnnl_dim_t)o.stride[1]};
    const dnnl_dim_t ks[2] = {(dnnl_dim_t)kernel[0], (dnnl_dim_t)kernel[1]};
    const dnnl_dim_t di[2] = {0, 0};
    const dnnl_dim_t pl[2] = {(dnnl_dim_t)o.pad[0], (dnnl_dim_t)o.pad[1]};
    const dnnl_dim_t pr[2] = {(dnnl_dim_t)o.pad[0], (dnnl_dim_t)o.pad[1]};
    Pd pd;
    if (dn::pool_pd(&pd.p, eng.p, dnnl_forward_inference,
                    (dnnl_alg_kind_t)(max_pool ? dnnl_pooling_max
                                               : dnnl_pooling_avg_exclude_padding),
                    ms.p, md.p, st, ks, di, pl, pr, nullptr) != dnnl_success) {
        return false;
    }
    return exec(eng.p, stm.p, pd.p, x.data<float>(), nullptr, y.data<float>());
}

// ---- 层归一(末维, affine 权重) ----
bool layer_norm(const Tensor& x, const Tensor& w, const Tensor& b,
                Tensor& y, double eps) {
    if (dn::family_off("norm") || !dn::available_onednn()) return false;
    Eng eng;
    Stm stm;
    if (!eng.ok() || !stm.ok()) return false;
    Md ms, mw, mb, md;
    if (!ms.from(x) || !mw.from(w) || !mb.from(b) || !md.from(y)) return false;
    Pd pd;
    if (dn::ln_pd(&pd.p, eng.p, dnnl_forward_inference, ms.p, md.p, nullptr,
                  (float)eps,
                  (unsigned)(dnnl_use_scale | dnnl_use_shift), nullptr) != dnnl_success) {
        return false;
    }
    Prim prim;
    if (dn::primitive_create(&prim.p, pd.p) != dnnl_success) return false;
    const_dnnl_memory_desc_t sp_md =
        dn::pd_query_md(pd.p, dnnl_query_scratchpad_md, 0);
    Mem m0, m1, m2, m3, msp;
    if (!m0.wrap(dn::pd_query_md(pd.p, dnnl_query_exec_arg_md, DNNL_ARG_SRC), eng.p, x.data<float>())) return false;
    if (!m1.wrap(dn::pd_query_md(pd.p, dnnl_query_exec_arg_md, DNNL_ARG_DST), eng.p, y.data<float>())) return false;
    if (!m2.wrap(dn::pd_query_md(pd.p, dnnl_query_exec_arg_md, DNNL_ARG_SCALE), eng.p, w.data<float>())) return false;
    if (!m3.wrap(dn::pd_query_md(pd.p, dnnl_query_exec_arg_md, DNNL_ARG_SHIFT), eng.p, b.data<float>())) return false;
    if (sp_md && !msp.wrap(sp_md, eng.p, DNNL_MEMORY_ALLOCATE)) return false;
    dnnl_exec_arg_t args[5] = {{DNNL_ARG_SRC, m0.p}, {DNNL_ARG_DST, m1.p},
                               {DNNL_ARG_SCALE, m2.p}, {DNNL_ARG_SHIFT, m3.p},
                               {DNNL_ARG_SCRATCHPAD, msp.p}};
    const int n = sp_md ? 5 : 4;
    if (dn::execute(prim.p, stm.p, n, args) != dnnl_success) return false;
    return dn::stream_wait(stm.p) == dnnl_success;
}

// ---- 批归一(推理语义: 用给定统计量) ----
bool batch_norm(const Tensor& x, const Tensor& w, const Tensor& b,
                const Tensor& mean, const Tensor& var, Tensor& y, double eps) {
    if (dn::family_off("norm") || !dn::available_onednn()) return false;
    Eng eng;
    Stm stm;
    if (!eng.ok() || !stm.ok()) return false;
    Md ms, md;
    if (!ms.from(x) || !md.from(y)) return false;
    Pd pd;
    if (dn::bn_pd(&pd.p, eng.p, dnnl_forward_inference, ms.p, md.p, (float)eps,
                  (unsigned)(dnnl_use_global_stats | dnnl_use_scale | dnnl_use_shift),
                  nullptr) != dnnl_success) {
        return false;
    }
    Prim prim;
    if (dn::primitive_create(&prim.p, pd.p) != dnnl_success) return false;
    const_dnnl_memory_desc_t sp_md =
        dn::pd_query_md(pd.p, dnnl_query_scratchpad_md, 0);
    Mem m0, m1, m2, m3, m4, m5, msp;
    auto q = [&](Mem& m, int arg, const void* p) {
        return m.wrap(dn::pd_query_md(pd.p, dnnl_query_exec_arg_md, arg), eng.p,
                      const_cast<void*>(p));
    };
    if (!q(m0, DNNL_ARG_SRC, x.data<float>())) return false;
    if (!q(m1, DNNL_ARG_DST, y.data<float>())) return false;
    if (!q(m2, DNNL_ARG_SCALE, w.data<float>())) return false;
    if (!q(m3, DNNL_ARG_SHIFT, b.data<float>())) return false;
    if (!q(m4, DNNL_ARG_MEAN, mean.data<float>())) return false;
    if (!q(m5, DNNL_ARG_VARIANCE, var.data<float>())) return false;
    if (sp_md && !msp.wrap(sp_md, eng.p, DNNL_MEMORY_ALLOCATE)) return false;
    dnnl_exec_arg_t args[7] = {{DNNL_ARG_SRC, m0.p}, {DNNL_ARG_DST, m1.p},
                               {DNNL_ARG_SCALE, m2.p}, {DNNL_ARG_SHIFT, m3.p},
                               {DNNL_ARG_MEAN, m4.p}, {DNNL_ARG_VARIANCE, m5.p},
                               {DNNL_ARG_SCRATCHPAD, msp.p}};
    const int n = sp_md ? 7 : 6;
    if (dn::execute(prim.p, stm.p, n, args) != dnnl_success) return false;
    return dn::stream_wait(stm.p) == dnnl_success;
}

} // namespace onednn
} // namespace znn
