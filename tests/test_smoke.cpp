// 基础测试: 创建/元信息/视图/拷贝/数据读取/引用计数语义
#include "znn.hpp"
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
