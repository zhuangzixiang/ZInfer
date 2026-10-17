// 基础测试: 创建/元信息/视图/拷贝/数据读取/引用计数语义
#include "znn.hpp"
#include "operator.hpp"

namespace znn {
extern const Operator g_matmul_operator;    // 对拍: 直接调用朴素 ref
extern const Operator g_conv2d_operator;
}
#include <cmath>
#include <string>
#include <cstdio>

using namespace znn;

static int failures = 0;

static void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

static void test_shape() {
    Shape s = {2, 3, 4};
    check(s.rank == 3 && s[0] == 2 && s[1] == 3 && s[2] == 4, "shape dims");
    check(s[-1] == 4 && s[-3] == 2, "shape negative index");
    check(Shape{}.rank == 0, "scalar shape rank 0");
}

static void test_create_and_read() {
    Tensor z = Tensor::zeros({2, 3});
    check(z.numel() == 6 && z.dim_count() == 2, "zeros numel/dim");
    bool all_zero = true;
    for (int64_t i = 0; i < 6; ++i) {
        all_zero = all_zero && z.data<float>()[i] == 0.f;
    }
    check(all_zero, "zeros value");
    Tensor o = Tensor::ones({3});
    check(o.data<float>()[2] == 1.f, "ones value");
    float src[4] = {1, 2, 3, 4};
    Tensor t = Tensor::from_host(src, {2, 2});
    check(t.data<float>()[0] == 1.f && t.data<float>()[3] == 4.f, "from_host");
    check(t.size(0) == 2 && t.size(-1) == 2, "size");
    Tensor one = Tensor::full({1}, 7.0);
    check(one.item<float>() == 7.f, "item value");
}

static void test_views() {
    float src[6] = {1, 2, 3, 4, 5, 6};
    Tensor t = Tensor::from_host(src, {2, 3});
    Tensor tr = t.transpose(0, 1);                        // {3,2} 非连续
    check(tr.size(0) == 3 && tr.size(1) == 2, "transpose shape");
    check(!tr.is_contiguous(), "transpose non-contiguous");
    check(tr.data<float>()[0] == 1.f, "transpose data base");
    Tensor c = tr.clone();                                // 紧凑化
    check(c.is_contiguous(), "clone contiguous");
    check(c.data<float>()[1] == 4.f && c.data<float>()[2] == 2.f, "clone layout");
    Tensor sl = t.slice(1, 1, 2);                         // 第 1 维取 [1,2) → {2,1}
    check(sl.size(0) == 2 && sl.size(1) == 1, "slice shape");
    Tensor sc = sl.clone();                                  // 视图先紧凑化再线性断言
    check(sc.data<float>()[0] == 2.f && sc.data<float>()[1] == 5.f, "slice values");
    Tensor fl = t.flatten();
    check(fl.numel() == 6 && fl.data<float>()[4] == 5.f, "flatten");
    Tensor sq = Tensor::from_host(src, {1, 6}).squeeze();
    check(sq.dim_count() == 1 && sq.numel() == 6, "squeeze");
    Tensor uq = sq.unsqueeze(0);
    check(uq.dim_count() == 2 && uq.size(0) == 1, "unsqueeze");
    Tensor pm = t.permute({1, 0});
    Tensor pc = pm.clone();
    check(pc.data<float>()[1] == 4.f, "permute");
    Tensor rs = t.reshape({3, 2});
    check(rs.data<float>()[2] == 3.f, "reshape");
}

static void test_lifetime() {
    Tensor a = Tensor::ones({4});
    Tensor b = a;                                         // 共享
    b.data<float>()[0] = 9.f;
    check(a.data<float>()[0] == 9.f, "handle shares storage");
    Tensor c;
    check(!static_cast<bool>(c), "null handle");
    a = Tensor::zeros({2});                               // 旧引用释放, b 仍有效
    check(b.data<float>()[3] == 1.f, "b alive after reassign");
}

static void test_grad_meta() {
    Tensor t = Tensor::ones({2, 2});
    t.requires_grad(true);
    check(t.requires_grad(), "requires_grad");
    t.grad().data<float>()[0] = 1.f;                      // 惰性分配
    check(t.grad().data<float>()[0] == 1.f, "grad persist");
    Tensor d = t.detach();
    check(!d.requires_grad() && d.data<float>()[1] == 1.f, "detach shares storage");
}

static void test_elementwise() {
    const float a[4] = {1, 2, 3, 4};
    const float b[4] = {10, 20, 30, 40};
    Tensor ta = Tensor::from_host(a, {2, 2});
    Tensor tb = Tensor::from_host(b, {2, 2});
    check(Tensor::add(ta, tb).data<float>()[3] == 44.f, "add");
    check(Tensor::sub(tb, ta).data<float>()[0] == 9.f, "sub");
    check(Tensor::mul(ta, tb).data<float>()[1] == 40.f, "mul");
    check(Tensor::div(tb, ta).data<float>()[2] == 10.f, "div");
    const float one[1] = {5};
    Tensor s = Tensor::add(ta, Tensor::from_host(one, {1}));  // 广播
    check(s.data<float>()[0] == 6.f && s.data<float>()[3] == 9.f, "broadcast add");
    const float negv[2] = {-1, 2};
    check(Activator::relu(Tensor::from_host(negv, {2})).data<float>()[0] == 0.f, "relu");
    Tensor sm = Activator::softmax(ta);
    const float s0 = sm.data<float>()[0] + sm.data<float>()[1];
    check(s0 > 0.999f && s0 < 1.001f, "softmax rows sum to 1");
}

static void test_autograd_chain() {
    Tensor x = Tensor::full({1}, 2.0);
    x.requires_grad(true);
    Tensor y = Tensor::mul(x, x);                       // y = x²
    Tensor z = Tensor::mul(y, y);                       // z = x⁴ = 16
    z.backward();
    check(z.item<float>() == 16.f, "autograd forward value");
    check(x.grad().item<float>() == 32.f, "chain rule dz/dx = 4x³");
}

static void test_matmul_reduce_loss() {
    const float a[4] = {1, 2, 3, 4};
    const float b[4] = {5, 6, 7, 8};
    Tensor A = Tensor::from_host(a, {2, 2});
    Tensor B = Tensor::from_host(b, {2, 2});
    Tensor M = Tensor::matmul(A, B);
    check(M.data<float>()[0] == 19.f && M.data<float>()[3] == 50.f, "matmul values");
    Tensor s = Tensor::sum(A, -1);
    check(s.shape().rank == 1 && s.data<float>()[0] == 3.f && s.data<float>()[1] == 7.f, "sum last dim");
    Tensor m = Tensor::mean(A, -1);
    check(m.data<float>()[0] == 1.5f, "mean");
    const float lg[4] = {2, 0, 0, 2};
    const int64_t tg[2] = {0, 1};
    Tensor loss = Loss::cross_entropy(Tensor::from_host(lg, {2, 2}),
                                      Tensor::from_host(tg, {2}, DType::I64));
    check(loss.item<float>() > 0.126f && loss.item<float>() < 0.128f, "cross entropy value");
    Tensor ms = Loss::mse(A, B);
    check(ms.item<float>() == 16.f, "mse value");   // 差恒为 -4, 均方 = 16
}

static void test_gradient_checks() {
    // 广播梯度: dl/dw = x
    const float xa[2] = {3, 4};
    Tensor x = Tensor::from_host(xa, {1, 2});
    Tensor w = Tensor::full({1, 2}, 2.0);
    w.requires_grad(true);
    Tensor y = Tensor::mul(x, w);
    Tensor l = Tensor::sum(y, -1);                     // {1}
    l.backward();
    check(w.grad().data<float>()[0] == 3.f && w.grad().data<float>()[1] == 4.f, "broadcast grad dw = x");

    // 有限差分校验: f(W) = ce(x @ W.T, y)
    const float xr[2] = {1, 2};
    const int64_t yr[1] = {0};
    Tensor X = Tensor::from_host(xr, {1, 2});
    Tensor Y = Tensor::from_host(yr, {1}, DType::I64);
    auto f = [&X, &Y](float w00, float w01) {
        const float w[2] = {w00, w01};
        return Loss::cross_entropy(Tensor::matmul(X, Tensor::from_host(w, {1, 2}).transpose(0, 1)), Y).item<float>();
    };
    const float w0[2] = {0.5f, -0.25f};
    Tensor W = Tensor::from_host(w0, {1, 2});
    W.requires_grad(true);
    Tensor logits = Tensor::matmul(X, W.transpose(0, 1));
    Tensor loss2 = Loss::cross_entropy(logits, Y);
    loss2.backward();
    const float e = 1e-3f;
    const float g0n = (f(w0[0] + e, w0[1]) - f(w0[0] - e, w0[1])) / (2 * e);
    const float g1n = (f(w0[0], w0[1] + e) - f(w0[0], w0[1] - e)) / (2 * e);
    const float g0a = W.grad().data<float>()[0];
    const float g1a = W.grad().data<float>()[1];
    check(fabsf(g0n - g0a) < 1e-2f && fabsf(g1n - g1a) < 1e-2f, "finite-diff grad check");
}

static void test_training_converges() {
    // 线性回归 y = 2x + 1, Adam 收敛到近似系数
    const int64_t N = 32;
    float xs[N], ys[N];
    for (int64_t i = 0; i < N; ++i) {
        xs[i] = static_cast<float>(i) / N - 0.5f;
        ys[i] = 2.f * xs[i] + 1.f;
    }
    Tensor X = Tensor::from_host(xs, {N, 1});
    Tensor Y = Tensor::from_host(ys, {N, 1});
    std::vector<Layer*> net = {
        Layer::create(LayerType::Linear, LayerParam{1, 1}),
    };
    Optimizer* opt = Optimizer::create(OptimType::Adam, net, OptimParam{0.05});
    float first = 0.f, last = 0.f;
    for (int step = 0; step < 400; ++step) {
        Tensor h = X;
        for (Layer* l : net) h = l->forward(h);
        Tensor loss = Loss::mse(h, Y);
        if (step == 0) first = loss.item<float>();
        last = loss.item<float>();
        opt->clear();
        loss.backward();
        opt->step();
    }
    Tensor h = X;
    for (Layer* l : net) h = l->forward(h);
    const float w = net[0]->params()[0].data<float>()[0];
    const float bias = net[0]->params()[1].data<float>()[0];
    check(last < first / 100.f, "training loss drops 100x");
    check(fabsf(w - 2.f) < 0.05f && fabsf(bias - 1.f) < 0.05f, "params converge to 2 and 1");
    Layer::destroy(net[0]);
    Optimizer::destroy(opt);
}


static void test_conv2d() {
    // 1x1x3x3 输入, 1x1x2x2 全 1 核, 无补零 → 2x2 窗口和
    const float x[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    const float w[4] = {1, 1, 1, 1};
    Tensor tx = Tensor::from_host(x, {1, 1, 3, 3});
    Tensor tw = Tensor::from_host(w, {1, 1, 2, 2});
    Tensor y = Tensor::conv2d(tx, tw);
    check(y.dim_count() == 4 && y.size(2) == 2 && y.size(3) == 2, "conv2d out shape");
    check(y.data<float>()[0] == 12.f && y.data<float>()[1] == 16.f
          && y.data<float>()[2] == 24.f && y.data<float>()[3] == 28.f, "conv2d values");

    // 有限差分校验卷积梯度(与 loss=sum(全部输出) 一致)
    auto f = [&tx](float w0) {
        const float wk[4] = {w0, 1, 1, 1};
        Tensor y = Tensor::conv2d(tx, Tensor::from_host(wk, {1, 1, 2, 2}));
        float s = 0.f;
        for (int64_t i = 0; i < y.numel(); ++i) s += y.data<float>()[i];
        return s;
    };
    Tensor W = Tensor::from_host(w, {1, 1, 2, 2});
    W.requires_grad(true);
    Tensor out = Tensor::conv2d(tx, W);
    Tensor loss = Tensor::sum(out.flatten(), -1);
    loss.backward();
    const float e = 1e-3f;
    const float gn = (f(1.f + e) - f(1.f - e)) / (2 * e);
    check(fabsf(gn - W.grad().data<float>()[0]) < 1e-2f, "conv2d fd grad check");
}

static void test_pool_cat_stack() {
    const float x[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor tx = Tensor::from_host(x, {1, 1, 2, 4});
    Pool2dOptions po;                                     // stride = kernel(非重叠池化)
    po.stride = {1, 2};
    Tensor mp = Tensor::max_pool2d(tx, {1, 2}, po);
    check(mp.size(3) == 2 && mp.data<float>()[0] == 2.f && mp.data<float>()[1] == 4.f, "max pool values");
    Tensor ap = Tensor::avg_pool2d(tx, {1, 2}, po);
    check(ap.data<float>()[0] == 1.5f && ap.data<float>()[1] == 3.5f, "avg pool values");

    // max pool 梯度只给窗口最大位置
    Tensor T = Tensor::from_host(x, {1, 1, 2, 4});
    T.requires_grad(true);
    Tensor l = Tensor::sum(Tensor::max_pool2d(T, {1, 2}, po).flatten(), -1);
    l.backward();
    check(T.grad().data<float>()[1] == 1.f && T.grad().data<float>()[0] == 0.f,
          "max pool grad routes to max");

    const float a[4] = {1, 2, 3, 4};
    const float b[4] = {5, 6, 7, 8};
    Tensor ta = Tensor::from_host(a, {2, 2});
    Tensor tb = Tensor::from_host(b, {2, 2});
    Tensor c0 = Tensor::cat(ta, tb, 0);
    check(c0.shape()[0] == 4 && c0.data<float>()[3] == 4.f && c0.data<float>()[4] == 5.f, "cat dim0");
    Tensor c1 = Tensor::cat(ta, tb, 1);
    check(c1.shape()[1] == 4 && c1.data<float>()[1] == 2.f && c1.data<float>()[2] == 5.f, "cat dim1");
    Tensor st = Tensor::stack({ta, tb}, 0);
    check(st.shape()[0] == 2 && st.data<float>()[4] == 5.f, "stack values");

    // cat 梯度按区间切片
    Tensor ra = Tensor::from_host(a, {2, 2});
    Tensor rb = Tensor::from_host(b, {2, 2});
    ra.requires_grad(true);
    rb.requires_grad(true);
    Tensor ls = Tensor::sum(Tensor::cat(ra, rb, 1).flatten(), -1);
    ls.backward();
    check(ra.grad().data<float>()[3] == 1.f && rb.grad().data<float>()[0] == 1.f, "cat grad slices");
}

static void test_embedding_reshape() {
    // embedding 取行
    const float w[6] = {1, 2, 3, 4, 5, 6};
    const int64_t idx[3] = {2, 0, 1};
    Tensor tw = Tensor::from_host(w, {3, 2});
    Tensor ti = Tensor::from_host(idx, {3}, DType::I64);
    Tensor e = Tensor::embedding(tw, ti);
    check(e.shape().rank == 2 && e.size(1) == 2 && e.data<float>()[0] == 5.f
          && e.data<float>()[2] == 1.f, "embedding gather");
    Tensor W = Tensor::from_host(w, {3, 2});
    W.requires_grad(true);
    Tensor le = Tensor::sum(Tensor::embedding(W, Tensor::from_host(idx, {3}, DType::I64)).flatten(), -1);
    le.backward();
    check(W.grad().data<float>()[4] == 1.f && W.grad().data<float>()[0] == 1.f, "embedding scatter grad");

    // reshape 训练态梯度回传: y = reshape(x{2,3} → {3,2}) 求 sum
    const float xr[6] = {1, 2, 3, 4, 5, 6};
    Tensor X = Tensor::from_host(xr, {2, 3});
    X.requires_grad(true);
    Tensor lr = Tensor::sum(X.reshape({3, 2}).flatten(), -1);
    lr.backward();
    bool all_one = true;
    for (int64_t i = 0; i < 6; ++i) all_one = all_one && X.grad().data<float>()[i] == 1.f;
    check(all_one, "reshape autograd identity grad");
}


static void test_norms_attention() {
    // layer_norm: 末维归一化, 输出每行均值≈偏置
    const float lx[4] = {1, 2, 3, 6};
    Tensor ln_x = Tensor::from_host(lx, {2, 2});
    LayerParam lp;
    lp.normalized_dim = 2;
    Layer* ln = Layer::create(LayerType::LayerNorm, lp);
    Tensor ly = ln->forward(ln_x);
    const float m0 = (ly.data<float>()[0] + ly.data<float>()[1]) / 2.f;
    check(fabsf(m0) < 1e-5f, "layer norm zero mean");
    Layer::destroy(ln);

    // batch_norm2d: 训练后输出每通道均值≈偏置(NCHW 布局正确索引)
    LayerParam bp;
    bp.channels = 2;
    Layer* bn = Layer::create(LayerType::BatchNorm2d, bp);
    const float bx[16] = {1, 10, 2, 11, 3, 12, 4, 13,
                          5, 20, 6, 21, 7, 22, 8, 23};
    Tensor bx_t = Tensor::from_host(bx, {2, 2, 2, 2});
    Tensor by = bn->forward(bx_t);
    float s0 = 0.f;
    for (int64_t n = 0; n < 2; ++n)
        for (int64_t hw = 0; hw < 4; ++hw)
            s0 += by.data<float>()[n * 8 + hw];             // channel 0: n*8 + 0*4 + h*2 + w
    check(fabsf(s0 / 8.f) < 1e-4f, "batch norm per-channel mean");
    Layer::destroy(bn);

    // attention: Q=单位阵, S=I/√2(非 I), softmax 后 P≠I, O≈加权平均 V
    const float q[4] = {1, 0, 0, 1};
    const float v[4] = {1, 2, 3, 4};
    Tensor tq = Tensor::from_host(q, {1, 1, 2, 2});
    Tensor tv = Tensor::from_host(v, {1, 1, 2, 2});
    Tensor att = Tensor::attention(tq, tq, tv);
    // S=[[0.707,0],[0,0.707]], softmax→P=[[0.67,0.33],[0.33,0.67]]
    // O = P@V → [0.67*1+0.33*3, 0.67*2+0.33*4, 0.33*1+0.67*3, 0.33*2+0.67*4]
    //        ≈ [1.66, 2.66, 2.34, 3.34]
    check(fabsf(att.data<float>()[0] - 1.66f) < 0.01f
          && fabsf(att.data<float>()[3] - 3.34f) < 0.01f, "attention identity q");
}
static void test_eigen_vs_ref() {
    // matmul: 派发(Eigen) vs 朴素 ref 数值一致
    Tensor a = Tensor::randn({17, 23}, 7);
    Tensor b = Tensor::randn({23, 11}, 8);
    Tensor fast = Tensor::matmul(a, b);
    Tensor slow = Tensor::empty(fast.shape());
    const Tensor* ins[2] = {&a, &b};
    Tensor* outs[1] = {&slow};
    KernelContext ctx{Device::CPU, nullptr};
    g_matmul_operator.ref(ctx, Span<const Tensor*>(ins, 2), Span<Tensor*>(outs, 1), nullptr);
    float max_diff = 0.f;
    for (int64_t i = 0; i < fast.numel(); ++i) {
        const float d = fabsf(fast.data<float>()[i] - slow.data<float>()[i]);
        max_diff = max_diff < d ? d : max_diff;
    }
    check(max_diff < 1e-3f, "matmul eigen vs ref allclose");

    // conv2d 同款对拍
    Tensor x = Tensor::randn({3, 4, 9, 9}, 9);
    Tensor w = Tensor::randn({6, 4, 3, 3}, 10);
    Conv2dOptions o;
    o.pad = {1, 1};
    Tensor f2 = Tensor::conv2d(x, w, o);
    Tensor s2 = Tensor::empty(f2.shape());
    const Tensor* ci[2] = {&x, &w};
    Tensor* co[1] = {&s2};
    g_conv2d_operator.ref(ctx, Span<const Tensor*>(ci, 2), Span<Tensor*>(co, 1), &o);
    float md2 = 0.f;
    for (int64_t i = 0; i < f2.numel(); ++i) {
        const float d = fabsf(f2.data<float>()[i] - s2.data<float>()[i]);
        md2 = md2 < d ? d : md2;
    }
    check(md2 < 1e-3f, "conv2d eigen vs ref allclose");
}


static void test_restriction_lifts() {
    // 任意维 sum/mean + 梯度
    const float a[6] = {1, 2, 3, 4, 5, 6};
    Tensor A = Tensor::from_host(a, {2, 3});
    Tensor s0 = Tensor::sum(A, 0);
    check(s0.dim_count() == 1 && s0.data<float>()[0] == 5.f && s0.data<float>()[2] == 9.f,
          "sum dim0 values");
    Tensor m1 = Tensor::mean(A, 1, true);
    check(m1.shape()[1] == 1 && fabsf(m1.data<float>()[0] - 2.f) < 1e-5f, "mean dim1 keepdim");
    Tensor R = Tensor::from_host(a, {2, 3});
    R.requires_grad(true);
    Tensor l0 = Tensor::sum(Tensor::sum(R, 0), -1);       // {1}
    l0.backward();
    check(R.grad().data<float>()[0] == 1.f && R.grad().data<float>()[5] == 1.f,
          "sum dim0 grad broadcast");

    // softmax 沿 dim 0: 每列和为 1
    Tensor sm0 = Activator::softmax(A, 0);
    const float col0 = sm0.data<float>()[0] + sm0.data<float>()[3];
    check(col0 > 0.999f && col0 < 1.001f, "softmax dim0 col sums 1");

    // permute 训练态: y = sum(permute(x, {1,0})) → dx 全 1
    Tensor P = Tensor::from_host(a, {2, 3});
    P.requires_grad(true);
    Tensor lp = Tensor::sum(P.permute({1, 0}).flatten(), -1);
    lp.backward();
    bool all1 = true;
    for (int64_t i = 0; i < 6; ++i) all1 = all1 && P.grad().data<float>()[i] == 1.f;
    check(all1, "permute training grad");

    // slice 训练态
    Tensor S = Tensor::zeros({3, 2});
    S.requires_grad(true);
    Tensor sv = S.slice(0, 1, 2);
    Tensor ls = Tensor::sum(sv.flatten(), -1);
    ls.backward();
    check(S.grad().data<float>()[0] == 0.f && S.grad().data<float>()[2] == 1.f
          && S.grad().data<float>()[3] == 1.f, "slice training grad scatters");
}

static void test_grouped_conv_and_causal() {
    // 分组卷积: groups=2 等价于两个独立半通道卷积拼接
    const float x[4] = {1, 2, 3, 4};                      // 1x4x1x1
    const float w[8] = {1, 1, 2, 2, 3, 3, 4, 4};           // 4x2x1x1, groups=2
    Tensor tx = Tensor::from_host(x, {1, 4, 1, 1});
    Tensor tw = Tensor::from_host(w, {4, 2, 1, 1});
    Conv2dOptions o;
    o.groups = 2;
    Tensor y = Tensor::conv2d(tx, tw, o);
    // 组0(ch0-1): o0=1·1+1·2=3, o1=2·1+2·2=6; 组1(ch2-3): o2=3·3+3·4=21, o3=4·3+4·4=28
    check(y.data<float>()[0] == 3.f && y.data<float>()[1] == 6.f
          && y.data<float>()[2] == 21.f && y.data<float>()[3] == 28.f, "grouped conv values");

    // causal attention: L=2, 未来位置被屏蔽 → 第 0 行输出只依赖 v[0]
    const float q[4] = {1, 0, 1, 0};
    const float v[4] = {10, 20, 30, 40};
    Tensor tq = Tensor::from_host(q, {1, 1, 2, 2});
    Tensor tv = Tensor::from_host(v, {1, 1, 2, 2});
    Tensor att = Tensor::attention(tq, tq, tv, true);
    check(fabsf(att.data<float>()[0] - 10.f) < 1e-4f
          && fabsf(att.data<float>()[1] - 20.f) < 1e-4f, "causal row0 attends only pos0");
}

static void test_dropout_layer() {
    LayerParam dp;
    dp.dropout_rate = 0.5;
    Layer* drop = Layer::create(LayerType::Dropout, dp);
    Tensor x = Tensor::ones({1000});
    x.requires_grad(true);
    Tensor y = drop->forward(x);                            // 训练态
    float kept = 0.f, scale_of_kept = 0.f;
    for (int64_t i = 0; i < 1000; ++i) {
        const float v = y.data<float>()[i];
        if (v > 0.f) { kept += 1.f; scale_of_kept = v; }
    }
    check(kept > 300.f && kept < 700.f, "dropout keeps ~half");
    check(fabsf(scale_of_kept - 2.f) < 1e-5f, "inverted dropout scale 1/(1-p)");
    Tensor loss = Tensor::sum(y, -1);
    loss.backward();
    // 梯度 = mask/(1-p) ∈ {0, 2}
    bool grad_ok = true;
    for (int64_t i = 0; i < 1000; ++i) {
        const float g = x.grad().data<float>()[i];
        grad_ok = grad_ok && (g == 0.f || g == 2.f);
    }
    check(grad_ok, "dropout grad is 0-or-2");
    drop->train(false);
    Tensor y2 = drop->forward(x);                           // eval 直通
    check(y2.data<float>()[0] == 1.f, "dropout eval passthrough");
    Layer::destroy(drop);
}


static void test_cuda_parity() {
    if (!Runtime::device_available(Device::CUDA)) {
        std::printf("CUDA not available, skipping GPU parity tests\n");
        return;
    }
    // 逐元素: GPU 与 CPU 结果一致
    Tensor a = Tensor::randn({64, 64}, 11);
    Tensor b = Tensor::randn({64, 64}, 12);
    Tensor cpu_add = Tensor::add(a, b);
    Tensor gpu_add = Tensor::add(a.to_device(Device::CUDA), b.to_device(Device::CUDA))
                         .to_device(Device::CPU);
    float max_d = 0.f;
    for (int64_t i = 0; i < cpu_add.numel(); ++i) {
        const float d = fabsf(cpu_add.data<float>()[i] - gpu_add.data<float>()[i]);
        max_d = max_d < d ? d : max_d;
    }
    check(max_d == 0.f, "cuda add parity");

    Tensor gpu_relu = Activator::relu(a.to_device(Device::CUDA)).to_device(Device::CPU);
    float rd = 0.f;
    for (int64_t i = 0; i < gpu_relu.numel(); ++i) {
        const float d = fabsf(Activator::relu(a).data<float>()[i] - gpu_relu.data<float>()[i]);
        rd = rd < d ? d : rd;
    }
    check(rd == 0.f, "cuda relu parity");

    // matmul 对拍: 切 FP32(与 CPU Eigen 逐位一致)
    Runtime::set_precision(Precision::FP32);
    Tensor m1 = Tensor::randn({32, 48}, 13);
    Tensor m2 = Tensor::randn({48, 24}, 14);
    Tensor cpu_mm = Tensor::matmul(m1, m2);
    Tensor gpu_mm = Tensor::matmul(m1.to_device(Device::CUDA), m2.to_device(Device::CUDA))
                        .to_device(Device::CPU);
    float md = 0.f;
    for (int64_t i = 0; i < cpu_mm.numel(); ++i) {
        const float d = fabsf(cpu_mm.data<float>()[i] - gpu_mm.data<float>()[i]);
        md = md < d ? d : md;
    }
    check(md < 1e-4f, "cuda matmul parity");

    // GEMM 类算子对拍(FP32): linear / conv2d / attention
    // (防"快而错": 曾有 ×w^T 朝向错误未被对拍覆盖)
    Tensor lx = Tensor::randn({16, 32}, 21);
    Tensor lw = Tensor::randn({24, 32}, 22);
    Tensor lb = Tensor::randn({24}, 23);
    Tensor cpu_lin = Tensor::linear(lx, lw, &lb);
    Tensor lb_g = lb.to_device(Device::CUDA);
    Tensor gpu_lin = Tensor::linear(lx.to_device(Device::CUDA),
                                    lw.to_device(Device::CUDA), &lb_g)
                         .to_device(Device::CPU);
    float ld = 0.f;
    for (int64_t i = 0; i < cpu_lin.numel(); ++i) {
        const float d = fabsf(cpu_lin.data<float>()[i] - gpu_lin.data<float>()[i]);
        ld = ld < d ? d : ld;
    }
    check(ld < 1e-4f, "cuda linear parity");

    Tensor cx = Tensor::randn({2, 6, 9, 9}, 31);
    Tensor cw = Tensor::randn({8, 6, 3, 3}, 32);
    Conv2dOptions co;
    co.pad = {1, 1};
    Tensor cpu_conv = Tensor::conv2d(cx, cw, co);
    Tensor gpu_conv = Tensor::conv2d(cx.to_device(Device::CUDA),
                                     cw.to_device(Device::CUDA), co)
                          .to_device(Device::CPU);
    float cd = 0.f;
    for (int64_t i = 0; i < cpu_conv.numel(); ++i) {
        const float d = fabsf(cpu_conv.data<float>()[i] - gpu_conv.data<float>()[i]);
        cd = cd < d ? d : cd;
    }
    check(cd < 1e-4f, "cuda conv2d parity");

    // attention: L/D 均 > 32(锁死旧内核越界 bug 的回归)
    Tensor q = Tensor::randn({2, 3, 48, 40}, 41);
    Tensor kk = Tensor::randn({2, 3, 48, 40}, 42);
    Tensor vv = Tensor::randn({2, 3, 48, 40}, 43);
    Tensor cpu_attn = Tensor::attention(q, kk, vv);
    Tensor gpu_attn = Tensor::attention(q.to_device(Device::CUDA),
                                        kk.to_device(Device::CUDA),
                                        vv.to_device(Device::CUDA))
                          .to_device(Device::CPU);
    float ad = 0.f;
    for (int64_t i = 0; i < cpu_attn.numel(); ++i) {
        const float d = fabsf(cpu_attn.data<float>()[i] - gpu_attn.data<float>()[i]);
        ad = ad < d ? d : ad;
    }
    check(ad < 1e-3f, "cuda attention parity");
    Runtime::set_precision(Precision::AUTO);

    // 兜底链: softmax 无 CUDA 内核 → GPU 输入自动落 CPU 计算再拷回
    Tensor sm = Activator::softmax(a.to_device(Device::CUDA));
    Tensor sm_cpu = sm.to_device(Device::CPU);
    check(sm_cpu.numel() == a.numel(), "cuda fallback softmax shape");

    // FP16 对拍: 半精度尾数 10 位, 相对误差容忍 2%
    Runtime::set_precision(Precision::FP16);
    Tensor f16_mm = Tensor::matmul(m1.to_device(Device::CUDA), m2.to_device(Device::CUDA))
                        .to_device(Device::CPU);
    float fd = 0.f;
    for (int64_t i = 0; i < cpu_mm.numel(); ++i) {
        const float r = fabsf(cpu_mm.data<float>()[i]);
        const float tol = 0.02f * (r > 1.f ? r : 1.f) + 1e-3f;
        const float d = fabsf(cpu_mm.data<float>()[i] - f16_mm.data<float>()[i]);
        fd = fd < d ? d : fd;
        check(d <= tol, "cuda fp16 matmul parity");
        if (d > tol) break;
    }
    Tensor cpu_lin_nb = Tensor::linear(lx, lw, nullptr);   // 同无 bias 参考
    Tensor f16_lin = Tensor::linear(lx.to_device(Device::CUDA),
                                    lw.to_device(Device::CUDA), nullptr)
                         .to_device(Device::CPU);
    for (int64_t i = 0; i < cpu_lin_nb.numel(); ++i) {
        const float r = fabsf(cpu_lin_nb.data<float>()[i]);
        const float tol = 0.02f * (r > 1.f ? r : 1.f) + 1e-3f;
        check(fabsf(cpu_lin_nb.data<float>()[i] - f16_lin.data<float>()[i]) <= tol,
              "cuda fp16 linear parity");
        if (failures) break;
    }
    Runtime::set_precision(Precision::AUTO);

    // GPU FP16 训练烟雾: 小 MLP + GradScaler, 损失须下降且全程有限
    {
        Runtime::set_precision(Precision::FP16);
        Tensor x = Tensor::randn({32, 8}, 51).to_device(Device::CUDA);
        Tensor t = Tensor::randn({32, 4}, 52).to_device(Device::CUDA);
        // 权重 ×0.1 初始化 + lr 3e-2: 稳定收敛配置(lrn 默认 1e-3 会发散,
        // 发散时 inf 跳步是 scaler 正常行为但无法验证"下降")
        Tensor w1 = Tensor::mul(Tensor::randn({16, 8}, 53),
                                Tensor::full(Shape{16, 8}, 0.1)).to_device(Device::CUDA);
        Tensor w2 = Tensor::mul(Tensor::randn({4, 16}, 54),
                                Tensor::full(Shape{4, 16}, 0.1)).to_device(Device::CUDA);
        w1.requires_grad(true);
        w2.requires_grad(true);
        OptimParam op;
        op.lr = 1e-1;
        Optimizer* opt = Optimizer::create(OptimType::Sgd,
                                           std::vector<Tensor>{w1, w2}, op);
        GradScaler scaler(256.f, 1000);
        float first = -1.f, last = -1.f;
        int good_steps = 0, inf_skips = 0;
        bool finite = true;
        for (int it = 0; it < 60; ++it) {
            Tensor y = Tensor::linear(
                Activator::relu(Tensor::linear(x, w1)), w2);
            Tensor loss = scaler.scale(Loss::mse(y, t));
            loss.backward();
            const bool ok = scaler.unscale(*opt);
            if (ok) { ++good_steps; opt->step(); } else { ++inf_skips; }
            scaler.update();
            opt->clear();
            const float lv = loss.to_device(Device::CPU).data<float>()[0];
            finite = finite && std::isfinite(lv);
            if (it == 0) first = lv;
            last = lv;
        }
        Optimizer::destroy(opt);
        check(finite && inf_skips < 30, "fp16 training finite (grad scaler)");
        check(good_steps >= 25, "fp16 training mostly steps");
        check(first - last > 0.03f, "fp16 training loss decreases");
        Runtime::set_precision(Precision::AUTO);
    }
    std::printf("CUDA parity tests passed\n");
}

// t21: GPU 反向全算子对拍 —— 同输入 CPU/GPU 双建图反传, 逐输入比梯度
// (铁律回归: 任何梯度规则的宿主循环/设备误用在此当场暴露)
static Tensor sum_all(const Tensor& t) {
    Tensor s = t;
    for (int64_t d = t.dim_count() - 1; d >= 0; --d) {
        s = Tensor::sum(s, d);
    }
    return s;
}

static void grad_parity(const char* name, std::vector<Tensor> inputs,
                        Tensor (*fwd)(const std::vector<Tensor>&),
                        float tol = 2e-3f) {
    std::vector<Tensor> xi = inputs, gi;
    for (Tensor& t : xi) {
        gi.push_back(t.to_device(Device::CUDA));
    }
    for (size_t i = 0; i < xi.size(); ++i) {
        xi[i].requires_grad(true);
        gi[i].requires_grad(true);
    }
    sum_all(fwd(xi)).backward();
    sum_all(fwd(gi)).backward();
    for (size_t i = 0; i < xi.size(); ++i) {
        if (xi[i].dtype() != DType::F32) continue;   // I64 等不可训练输入无梯度语义
        Tensor gc = xi[i].grad().to_device(Device::CPU);
        Tensor gg = gi[i].grad().to_device(Device::CPU);
        float md = 0.f;
        for (int64_t k = 0; k < gc.numel(); ++k) {
            const float d = fabsf(gc.data<float>()[k] - gg.data<float>()[k]);
            md = md < d ? d : md;
        }
        check(md < tol, name);   // 失败时按算子名定位, maxerr 见断言消息
    }
}

static Tensor fwd_binary(const std::vector<Tensor>& v) { return Tensor::mul(Tensor::add(v[0], v[1]), Tensor::sub(v[0], v[1])); }
static Tensor fwd_unary(const std::vector<Tensor>& v) { return Activator::gelu(Activator::sigmoid(v[0])); }
static Tensor fwd_reduce(const std::vector<Tensor>& v) { return Tensor::sum(Tensor::mean(v[0], 1, true), 0); }
static Tensor fwd_matmul(const std::vector<Tensor>& v) { return Tensor::matmul(v[0], v[1]); }
static Tensor fwd_linear(const std::vector<Tensor>& v) { return Tensor::linear(v[0], v[1], &v[2]); }
static Tensor fwd_conv(const std::vector<Tensor>& v) { return Tensor::conv2d(v[0], v[1]); }
static Tensor fwd_pool(const std::vector<Tensor>& v) {
    return Tensor::max_pool2d(Tensor::avg_pool2d(v[0], {2, 2}), {2, 2});
}
static Tensor fwd_embed(const std::vector<Tensor>& v) { return Tensor::embedding(v[0], v[1]); }
static Tensor fwd_softmax(const std::vector<Tensor>& v) { return Activator::softmax(v[0]); }
static Tensor fwd_attn(const std::vector<Tensor>& v) { return Tensor::attention(v[0], v[1], v[2]); }
static Tensor fwd_loss(const std::vector<Tensor>& v) { return Tensor::mul(Loss::cross_entropy(v[0], v[1]), Loss::mse(v[2], v[3])); }
static Tensor fwd_views(const std::vector<Tensor>& v) {
    Tensor c = Tensor::cat(v[0], v[1], 1);                       // [2,8]
    Tensor p = c.permute({1, 0});                                // [8,2]
    return Tensor::matmul(p, v[0].slice(1, 0, 1));               // [8,1]
}

static void test_gpu_backward_parity() {
    if (!Runtime::device_available(Device::CUDA)) {
        std::printf("CUDA not available, skipping GPU backward parity\n");
        return;
    }
    Runtime::set_precision(Precision::FP32);    // 对拍用位精确 FP32
    grad_parity("binary", {Tensor::randn({6, 5}, 71), Tensor::randn({6, 5}, 72)}, fwd_binary);
    grad_parity("unary", {Tensor::randn({6, 5}, 73)}, fwd_unary);
    grad_parity("reduce", {Tensor::randn({4, 6, 5}, 74)}, fwd_reduce);
    grad_parity("matmul", {Tensor::randn({5, 7}, 75), Tensor::randn({7, 4}, 76)}, fwd_matmul);
    grad_parity("linear", {Tensor::randn({5, 7}, 77), Tensor::randn({3, 7}, 78), Tensor::randn({3}, 79)}, fwd_linear);
    grad_parity("conv2d", {Tensor::randn({2, 3, 7, 7}, 80), Tensor::randn({4, 3, 3, 3}, 81)}, fwd_conv);
    grad_parity("pool", {Tensor::randn({2, 3, 8, 8}, 82)}, fwd_pool);
    float wdata[10] = {0.f, 1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 1.f};
    int64_t idata[6] = {0, 2, 4, 1, 3, 0};      // indices 为 I64 且须在词表界内
    grad_parity("embedding", {Tensor::from_host(wdata, {5, 2}),
                              Tensor::from_host(idata, {2, 3}, DType::I64)},
                fwd_embed, 1e-4f);
    grad_parity("softmax", {Tensor::randn({4, 6}, 83)}, fwd_softmax);
    // LN/BN 经 Layer API 端到端对拍(前向输出 + 输入梯度, CPU/GPU)
    {
        LayerParam lp;
        lp.normalized_dim = 8;   // 末维大小(PyTorch LayerNorm(N) 语义)
        Layer* ln = Layer::create(LayerType::LayerNorm, lp);
        Tensor xc = Tensor::randn({3, 4, 8}, 84);
        Tensor xg = xc.to_device(Device::CUDA);
        xc.requires_grad(true);
        xg.requires_grad(true);
        Tensor yc = ln->forward(xc);
        Tensor yg = ln->forward(xg);
        sum_all(yc).backward();
        sum_all(yg).backward();
        float od = 0.f, gd = 0.f;
        Tensor ygc = yg.to_device(Device::CPU);
        Tensor ggc = xg.grad().to_device(Device::CPU);
        for (int64_t k = 0; k < yc.numel(); ++k) {
            od = fmaxf(od, fabsf(yc.data<float>()[k] - ygc.data<float>()[k]));
            gd = fmaxf(gd, fabsf(xc.grad().data<float>()[k] - ggc.data<float>()[k]));
        }
        Layer::destroy(ln);
        check(od < 1e-4f && gd < 1e-3f, "gpu bwd[layer_norm]");
    }
    grad_parity("attention", {Tensor::randn({1, 2, 6, 4}, 86),
                              Tensor::randn({1, 2, 6, 4}, 87),
                              Tensor::randn({1, 2, 6, 4}, 88)}, fwd_attn, 5e-3f);
    Tensor lg = Tensor::randn({5, 4}, 89);
    int64_t tg[5] = {0, 1, 2, 3, 0};           // CE 标签须为 I64
    grad_parity("losses", {lg, Tensor::from_host(tg, {5}, DType::I64),
                           Tensor::randn({5, 3}, 90),
                           Tensor::randn({5, 3}, 91)}, fwd_loss);
    grad_parity("views", {Tensor::randn({2, 4}, 92), Tensor::randn({2, 4}, 93)}, fwd_views);
    Runtime::set_precision(Precision::AUTO);
    std::printf("GPU backward parity tests passed\n");
}

// t20: GradScaler 动态损失缩放(机制单元, CPU 路径)
static void test_grad_scaler() {
    GradScaler scaler(1024.f, 2);
    Tensor x = Tensor::randn({5, 4}, 62);
    Tensor t = Tensor::randn({5, 3}, 63);
    Tensor w = Tensor::randn({3, 4}, 61);
    w.requires_grad(true);
    Optimizer* opt = Optimizer::create(OptimType::Sgd,
                                       std::vector<Tensor>{w}, OptimParam{});

    // 参考梯度(无缩放; 标量损失才能 backward)
    Loss::mse(Tensor::linear(x, w), t).backward();
    Tensor ref_g = w.grad();
    w.grad() = Tensor::zeros(w.shape());                    // 清梯度(直接覆盖)
    // 缩放路径: 新前向建图, 反传后梯度 = 原梯度 × s
    scaler.scale(Loss::mse(Tensor::linear(x, w), t)).backward();
    float ratio_err = 0.f;
    for (int64_t i = 0; i < w.numel(); ++i) {
        const float g0 = ref_g.data<float>()[i];
        const float g1 = w.grad().data<float>()[i];
        if (fabsf(g0) > 1e-6f) {
            ratio_err = fmaxf(ratio_err, fabsf(g1 / scaler.scale_factor() - g0) /
                                           (fabsf(g0) + 1e-6f));
        }
    }
    check(ratio_err < 1e-4f, "scaler scale multiplies grads");

    // unscale: 有限梯度 → true 且梯度恢复原值
    check(scaler.unscale(*opt), "unscale finite");
    float back_err = 0.f;
    for (int64_t i = 0; i < w.numel(); ++i) {
        const float d = fabsf(w.grad().data<float>()[i] - ref_g.data<float>()[i]);
        back_err = back_err < d ? d : back_err;
    }
    check(back_err < 1e-3f, "unscale restores grads");

    // inf 梯度 → false; update 收缩 scale
    w.grad().data<float>()[0] = INFINITY;
    check(!scaler.unscale(*opt), "unscale detects inf");
    scaler.update();
    check(scaler.scale_factor() == 512.f, "update halves scale");
    w.grad().data<float>()[0] = 1.f;         // 复位毒化值, 进入健康期
    // 连续健康 → 每 growth_interval 步 ×2
    scaler.unscale(*opt);
    scaler.update();
    scaler.unscale(*opt);
    scaler.update();
    check(scaler.scale_factor() == 1024.f, "update grows scale");
    Optimizer::destroy(opt);
}

int main() {
    std::fprintf(stderr, "t1\n"); test_shape();
    std::fprintf(stderr, "t2\n"); test_create_and_read();
    std::fprintf(stderr, "t3\n"); test_views();
    std::fprintf(stderr, "t4\n"); test_lifetime();
    std::fprintf(stderr, "t5\n"); test_grad_meta();
    std::fprintf(stderr, "t6\n"); test_elementwise();
    std::fprintf(stderr, "t7\n"); test_autograd_chain();
    std::fprintf(stderr, "t8\n"); test_matmul_reduce_loss();
    std::fprintf(stderr, "t9\n"); test_gradient_checks();
    std::fprintf(stderr, "t10\n"); test_training_converges();
    std::fprintf(stderr, "t11\n"); test_conv2d();
    std::fprintf(stderr, "t12\n"); test_pool_cat_stack();
    std::fprintf(stderr, "t13\n"); test_embedding_reshape();
    std::fprintf(stderr, "t14\n"); test_norms_attention();
    std::fprintf(stderr, "t15\n"); test_eigen_vs_ref();
    std::fprintf(stderr, "t16\n"); test_restriction_lifts();
    std::fprintf(stderr, "t17\n"); test_grouped_conv_and_causal();
    std::fprintf(stderr, "t18\n"); test_dropout_layer();
    std::fprintf(stderr, "t19\n"); test_cuda_parity();
    std::fprintf(stderr, "t20\n"); test_grad_scaler();
    std::fprintf(stderr, "t21\n"); test_gpu_backward_parity();
    if (failures == 0) {
        std::printf("ALL BASIC TESTS PASSED\n");
        return 0;
    }
    std::printf("%d FAILURES\n", failures);
    return 1;
}
