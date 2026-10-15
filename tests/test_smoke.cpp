// 基础测试: 创建/元信息/视图/拷贝/数据读取/引用计数语义
#include "znn.hpp"
#include "operator.hpp"

namespace znn {
extern const Operator g_matmul_operator;    // 对拍: 直接调用朴素 ref
extern const Operator g_conv2d_operator;
}
#include <cmath>
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
    std::fprintf(stderr, "setup1\n");
    std::vector<Layer*> net = {
        Layer::create(LayerType::Linear, LayerParam{1, 1}),
    };
    std::fprintf(stderr, "setup2\n");
    Optimizer* opt = Optimizer::create(OptimType::Adam, net, OptimParam{0.05});
    std::fprintf(stderr, "setup3\n");
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
        std::fprintf(stderr, "o\n");
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

    // 有限差分校验卷积梯度
    auto f = [&tx](float w0) {
        const float wk[4] = {w0, 1, 1, 1};
        return Tensor::conv2d(tx, Tensor::from_host(wk, {1, 1, 2, 2}))
                   .data<float>()[0];
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
    Tensor mp = Tensor::max_pool2d(tx, {1, 2});
    check(mp.size(3) == 2 && mp.data<float>()[0] == 2.f && mp.data<float>()[1] == 4.f, "max pool values");
    Tensor ap = Tensor::avg_pool2d(tx, {1, 2});
    check(ap.data<float>()[0] == 1.5f && ap.data<float>()[1] == 3.5f, "avg pool values");

    // max pool 梯度只给窗口最大位置
    Tensor T = Tensor::from_host(x, {1, 1, 2, 4});
    T.requires_grad(true);
    Tensor l = Tensor::sum(Tensor::max_pool2d(T, {1, 2}).flatten(), -1);
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
    check(ra.grad().data<float>()[3] == 4.f && rb.grad().data<float>()[0] == 5.f, "cat grad slices");
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
    Layer* ln = Layer::create(LayerType::LayerNorm, LayerParam{2});
    Tensor ly = ln->forward(ln_x);
    const float m0 = (ly.data<float>()[0] + ly.data<float>()[1]) / 2.f;
    check(fabsf(m0) < 1e-5f, "layer norm zero mean");
    Layer::destroy(ln);

    // batch_norm2d: 训练后输出每通道均值≈偏置
    LayerParam bp;
    bp.channels = 2;
    Layer* bn = Layer::create(LayerType::BatchNorm2d, bp);
    const float bx[16] = {1, 10, 2, 11, 3, 12, 4, 13,
                          5, 20, 6, 21, 7, 22, 8, 23};
    Tensor bx_t = Tensor::from_host(bx, {2, 2, 2, 2});
    Tensor by = bn->forward(bx_t);
    float s0 = 0.f;
    for (int64_t i = 0; i < 8; ++i) s0 += by.data<float>()[i * 2];
    check(fabsf(s0 / 8.f) < 1e-4f, "batch norm per-channel mean");
    Layer::destroy(bn);

    // attention: B=1,H=1,L=2,D=2 手算
    const float q[4] = {1, 0, 0, 1};
    const float v[4] = {1, 2, 3, 4};
    Tensor tq = Tensor::from_host(q, {1, 1, 2, 2});
    Tensor tv = Tensor::from_host(v, {1, 1, 2, 2});
    Tensor att = Tensor::attention(tq, tq, tv);
    // S = I → P = I → O = V
    check(att.data<float>()[0] == 1.f && att.data<float>()[3] == 4.f, "attention identity q");
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

    // slice 训练态: 取行 1, loss=sum → 只有该行有梯度
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
    const float x[8] = {1, 2, 3, 4, 5, 6, 7, 8};           // 1x4x2x2? 用 1x4x1x2? 用 {1,4,1,1}
    const float w[8] = {1, 1, 2, 2, 3, 3, 4, 4};           // 4x1x1x1, groups=2
    Tensor tx = Tensor::from_host(x, {1, 4, 1, 1});
    Tensor tw = Tensor::from_host(w, {4, 1, 1, 1});
    Conv2dOptions o;
    o.groups = 2;
    Tensor y = Tensor::conv2d(tx, tw, o);
    // 每通道 = 权重·对应输入: [1,4,6,12]
    check(y.data<float>()[0] == 1.f && y.data<float>()[1] == 4.f
          && y.data<float>()[2] == 9.f && y.data<float>()[3] == 12.f, "grouped conv values");

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
    std::fprintf(stderr, "done-tests\n");
    if (failures == 0) {
        std::printf("ALL BASIC TESTS PASSED\n");
        return 0;
    }
    std::printf("%d FAILURES\n", failures);
    return 1;
}
