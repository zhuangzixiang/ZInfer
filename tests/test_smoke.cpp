// 基础测试: 创建/元信息/视图/拷贝/数据读取/引用计数语义
#include "znn.hpp"
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

int main() {
    test_shape();
    test_create_and_read();
    test_views();
    test_lifetime();
    test_grad_meta();
    test_elementwise();
    test_autograd_chain();
    if (failures == 0) {
        std::printf("ALL BASIC TESTS PASSED\n");
        return 0;
    }
    std::printf("%d FAILURES\n", failures);
    return 1;
}
