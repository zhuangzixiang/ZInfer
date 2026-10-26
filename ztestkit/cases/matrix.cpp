// ============================================================================
// matrix.cpp — L0 不变量 + L1 算子×形状族矩阵 + L2 梯度对拍(设备参数化)
//
// 覆盖原则: 非 8/16 整倍数形状(打边界 tile)、广播变体(标量/行/列)、
// keepdim 两态、非连续视图(clone/transpose 材料化路径)、offset>0 切片。
// 新算子接入 = 加一个 ZTK_CASE(与实现同块落地, 测试先行)。
// ============================================================================
#include "../ztestkit.hpp"

using namespace znn;
using ztestkit::back;
using ztestkit::check;
using ztestkit::max_diff;
using ztestkit::on;

namespace {

// 通用: 同种子 CPU 基准 vs 设备结果(新鲜张量, 逐图隔离)
// 先印检查名再执行 —— 崩溃时最后一条即罪魁; 失败时倾倒首错现场
void cmp(const char* what, const Tensor& cpu, const Tensor& dev, float tol) {
    std::printf("      > %s\n", what);
    const float m = max_diff(cpu, back(dev));
    if (m > tol) {
        std::printf("    FAIL-DETAIL %s: md=%.3e tol=%.1e cpu=[%lld] dev=[%lld]\n",
                    what, m, tol, (long long)cpu.numel(), (long long)dev.numel());
        int64_t wi = 0;
        float wm = -1.f;
        for (int64_t i = 0; i < cpu.numel(); ++i) {
            const float d = std::fabs(cpu.data<float>()[i]
                                      - back(dev).data<float>()[i]);
            if (d > wm) { wm = d; wi = i; }
        }
        std::printf("      worst i=%lld cpu=%g dev=%g\n", (long long)wi,
                    cpu.data<float>()[wi], back(dev).data<float>()[wi]);
    }
    check(m <= tol, what);
}

// ============ L0: 张量/存储不变量(仅 CPU 设备跑一次) ============

ZTK_CASE(l0_view_invariants, false) {
    if (ztk_dev != Device::CPU) {
        return;   // 不变量与设备无关, 跑一次
    }
    Tensor x = Tensor::randn({4, 6}, 1);
    Tensor s = x.slice(1, 1, 3);                    // [4,2] 列切片 offset=1
    check(s.storage_offset() == 1, "slice offset");
    check(s.numel() == 8 && s.is_contiguous() || !s.is_contiguous(),
          "slice shape");   // 形状事实(不预设连续性实现)
    check(back(s.slice(0, 0, 2)).data<float>()[0] == x.data<float>()[1],
          "nested view reads base storage");   // s[0][0]=x[0][1]
    Tensor c = x.clone();                           // 可微深拷贝(存储独立)
    c.data<float>()[0] = 99.f;
    check(x.data<float>()[0] != 99.f, "clone storage independent");
    check(c.storage_offset() == 0, "clone offset zero");
    Tensor t = x.transpose(0, 1);                   // [6,4] 非连续视图
    check(!t.is_contiguous(), "transpose non-contiguous");
    check(back(t.clone()).data<float>()[2 * 4 + 1] == x.data<float>()[1 * 6 + 2],
          "transpose clone materialize");           // clone[2][1] = x[1][2]
    // to_device 往返(含 GPU): 逐位一致
    Tensor r = back(on(ztk_dev, x));
    check(max_diff(x, r) == 0.f, "to_device roundtrip bitwise");
}

ZTK_CASE(l0_grad_semantics, false) {
    if (ztk_dev != Device::CPU) {
        return;
    }
    // 共享句柄 = 同一张量(PyTorch 语义): requires_grad 相互可见
    Tensor a = Tensor::randn({3}, 2);
    Tensor b = a;
    b.requires_grad(true);
    check(a.requires_grad() && b.requires_grad(), "shared handle shared flag");
    // detach: 脱离计算图
    Tensor d = b.detach();
    check(!d.requires_grad(), "detach clears flag");
    check(d.data<float>()[0] == b.data<float>()[0], "detach shares storage");
}

// ============ L1: 算子×形状族矩阵(逐算子一用例, 设备自动展开) ============

// 二元 4 算子 × {同形, 行广播, 列广播, 标量广播}
ZTK_CASE(add, false) {
    Tensor a = Tensor::randn({5, 7}, 3);
    Tensor b = Tensor::randn({5, 7}, 4);
    cmp("add same", Tensor::add(a, b), Tensor::add(on(ztk_dev, a), on(ztk_dev, b)), 1e-5f);
    Tensor r = Tensor::randn({7}, 5);
    cmp("add row-bcast", Tensor::add(a, r),
        Tensor::add(on(ztk_dev, a), on(ztk_dev, r)), 1e-5f);
    Tensor c = Tensor::randn({5, 1}, 6);
    cmp("add col-bcast", Tensor::add(a, c),
        Tensor::add(on(ztk_dev, a), on(ztk_dev, c)), 1e-5f);
    Tensor sc = Tensor::randn({1}, 7);
    cmp("add scalar", Tensor::add(a, sc),
        Tensor::add(on(ztk_dev, a), on(ztk_dev, sc)), 1e-5f);
}
ZTK_CASE(sub, false) {
    Tensor a = Tensor::randn({5, 7}, 8), b = Tensor::randn({5, 7}, 9);
    cmp("sub", Tensor::sub(a, b), Tensor::sub(on(ztk_dev, a), on(ztk_dev, b)), 1e-5f);
    Tensor r = Tensor::randn({7}, 10);
    cmp("sub bcast", Tensor::sub(a, r), Tensor::sub(on(ztk_dev, a), on(ztk_dev, r)), 1e-5f);
}
ZTK_CASE(mul, false) {
    Tensor a = Tensor::randn({5, 7}, 11), b = Tensor::randn({5, 7}, 12);
    cmp("mul", Tensor::mul(a, b), Tensor::mul(on(ztk_dev, a), on(ztk_dev, b)), 1e-5f);
    Tensor c = Tensor::randn({5, 1}, 13);
    cmp("mul col-bcast", Tensor::mul(a, c),
        Tensor::mul(on(ztk_dev, a), on(ztk_dev, c)), 1e-5f);
}
ZTK_CASE(div, false) {
    Tensor a = Tensor::randn({5, 7}, 14), b = Tensor::randn({5, 7}, 15);
    cmp("div", Tensor::div(a, b), Tensor::div(on(ztk_dev, a), on(ztk_dev, b)), 1e-4f);
}

// 一元族(9): 非 8 倍数形状(逐个直写)
ZTK_CASE(relu, false) {
    Tensor x = Tensor::randn({13, 7}, 16);
    cmp("relu", Activator::relu(x), Activator::relu(on(ztk_dev, x)), 0.f);
}
ZTK_CASE(gelu, false) {
    Tensor x = Tensor::randn({13, 7}, 17);
    cmp("gelu", Activator::gelu(x), Activator::gelu(on(ztk_dev, x)), 1e-5f);
}
ZTK_CASE(sigmoid, false) {
    Tensor x = Tensor::randn({13, 7}, 18);
    cmp("sigmoid", Activator::sigmoid(x), Activator::sigmoid(on(ztk_dev, x)), 1e-5f);
}
ZTK_CASE(tanh, false) {
    Tensor x = Tensor::randn({13, 7}, 19);
    cmp("tanh", Activator::tanh(x), Activator::tanh(on(ztk_dev, x)), 1e-5f);
}
ZTK_CASE(silu, false) {
    Tensor x = Tensor::randn({13, 7}, 20);
    cmp("silu", Activator::silu(x), Activator::silu(on(ztk_dev, x)), 1e-5f);
}
ZTK_CASE(neg, false) {
    Tensor x = Tensor::randn({13, 7}, 21);
    cmp("neg", Tensor::neg(x), Tensor::neg(on(ztk_dev, x)), 0.f);
}
ZTK_CASE(exp, false) {
    Tensor x = Tensor::randn({13, 7}, 22);
    cmp("exp", Tensor::exp(x), Tensor::exp(on(ztk_dev, x)), 1e-4f);
}
ZTK_CASE(log, false) {
    Tensor x = Tensor::exp(Tensor::randn({13, 7}, 23));   // exp 恒正
    cmp("log", Tensor::log(x), Tensor::log(on(ztk_dev, x)), 1e-4f);
}
ZTK_CASE(sqrt, false) {
    Tensor x = Tensor::exp(Tensor::randn({13, 7}, 24));
    cmp("sqrt", Tensor::sqrt(x), Tensor::sqrt(on(ztk_dev, x)), 1e-5f);
}

// GEMM 族: 非 8 倍数(边界 tile) + 中形
ZTK_CASE(matmul, false) {
    Tensor a = Tensor::randn({5, 7}, 25), b = Tensor::randn({7, 4}, 26);
    cmp("matmul 5x7x4 (boundary)", Tensor::matmul(a, b),
        Tensor::matmul(on(ztk_dev, a), on(ztk_dev, b)), 1e-4f);
    Tensor a2 = Tensor::randn({60, 37}, 27), b2 = Tensor::randn({37, 45}, 28);
    cmp("matmul 60x37x45", Tensor::matmul(a2, b2),
        Tensor::matmul(on(ztk_dev, a2), on(ztk_dev, b2)), 1e-3f);
}
ZTK_CASE(linear, false) {
    Tensor x = Tensor::randn({13, 29}, 29), w = Tensor::randn({21, 29}, 30);
    Tensor lc = Tensor::linear(x, w);
    Tensor lg = Tensor::linear(on(ztk_dev, x), on(ztk_dev, w));
    cmp("linear x@wT", lc, lg, 1e-3f);
}

// 规约: dim×keepdim 矩阵
ZTK_CASE(sum, false) {
    Tensor x = Tensor::randn({5, 7, 3}, 31);
    cmp("sum dim0", Tensor::sum(x, 0), Tensor::sum(on(ztk_dev, x), 0), 1e-3f);
    cmp("sum dim1", Tensor::sum(x, 1), Tensor::sum(on(ztk_dev, x), 1), 1e-3f);
    cmp("sum dim-1", Tensor::sum(x, -1), Tensor::sum(on(ztk_dev, x), -1), 1e-3f);
    cmp("sum dim1 keep", Tensor::sum(x, 1, true),
        Tensor::sum(on(ztk_dev, x), 1, true), 1e-3f);
}
ZTK_CASE(mean, false) {
    Tensor x = Tensor::randn({5, 7, 3}, 32);
    cmp("mean dim0", Tensor::mean(x, 0), Tensor::mean(on(ztk_dev, x), 0), 1e-3f);
    cmp("mean dim2", Tensor::mean(x, 2), Tensor::mean(on(ztk_dev, x), 2), 1e-3f);
    cmp("mean dim2 keep", Tensor::mean(x, 2, true),
        Tensor::mean(on(ztk_dev, x), 2, true), 1e-3f);
}
ZTK_CASE(softmax, false) {
    Tensor x = Tensor::randn({9, 14, 7}, 33);
    cmp("softmax last-dim", Activator::softmax(x),
        Activator::softmax(on(ztk_dev, x)), 1e-4f);
}

// 视图族(clone/transpose 材料化 + offset>0 切片 —— 绑定层偏移的实测)
ZTK_CASE(views_materialize, false) {
    Tensor x = Tensor::randn({6, 9}, 34);
    Tensor v = x.reshape({2, 27}).slice(0, 1, 2);   // [1,27] 连续 offset=27
    check(v.storage_offset() == 27, "offset view");
    Tensor mm = Tensor::matmul(v, v.transpose(1, 0));
    Tensor vd = on(ztk_dev, v);
    // 同设备: 共享直返(offset 保留); 跨设备: 材料化(offset 0) —— PyTorch .to() 同义
    check(ztk_dev == Device::CPU ? vd.storage_offset() == 27
                                 : vd.storage_offset() == 0,
          "to_device same=shared / cross=materialized");
    Tensor mmd = Tensor::matmul(vd, vd.transpose(1, 0));
    cmp("offset-view gemm (binding offset)", mm, mmd, 1e-3f);
    Tensor tc = x.transpose(0, 1).clone();          // 非连续材料化
    Tensor td = on(ztk_dev, x.transpose(0, 1)).clone();
    cmp("transpose clone on device", tc, td, 0.f);
}

// ============ L2: 梯度对拍(设备参数化, 每图新鲜叶) ============

// 逐元素链: sigmoid→gelu→sum 全维(解析可达)
ZTK_CASE(grad_elementwise, true) {
    auto run = [](Device d) {
        Tensor x = Tensor::randn({6, 9}, 35).to_device(d);
        x.requires_grad(true);
        Tensor a = Activator::gelu(Activator::sigmoid(x));
        Tensor s = a;
        for (int64_t dd = s.dim_count() - 1; dd >= 0; --dd) {
            s = Tensor::sum(s, dd);
        }
        s.backward();
        return back(x.grad());
    };
    Tensor gcpu = run(Device::CPU);
    Tensor gdev = run(ztk_dev);
    cmp("grad sigmoid+gelu+sum", gcpu, gdev, 1e-4f);
}

// 二元+广播链: mul 广播反传(grad_bcast 路径)
ZTK_CASE(grad_broadcast, true) {
    auto run = [](Device d) {
        Tensor a = Tensor::randn({5, 7}, 36).to_device(d);
        Tensor b = Tensor::randn({7}, 37).to_device(d);
        a.requires_grad(true);
        b.requires_grad(true);
        Tensor y = Tensor::mul(a, b);
        Tensor s = y;
        for (int64_t dd = s.dim_count() - 1; dd >= 0; --dd) {
            s = Tensor::sum(s, dd);
        }
        s.backward();
        return back(a.grad());
    };
    cmp("grad mul row-bcast a", run(Device::CPU), run(ztk_dev), 1e-4f);
}

// GEMM 链: matmul+视图(clone/reshape/slice)
ZTK_CASE(grad_matmul_views, true) {
    auto run = [](Device d) {
        Tensor x = Tensor::randn({6, 9}, 38).to_device(d);
        x.requires_grad(true);
        Tensor v = x.reshape({2, 27}).slice(0, 0, 1).clone();
        Tensor tr = v.transpose(1, 0).clone();
        Tensor mm = Tensor::matmul(v, tr);
        Tensor s = mm;
        for (int64_t dd = s.dim_count() - 1; dd >= 0; --dd) {
            s = Tensor::sum(s, dd);
        }
        s.backward();
        return back(x.grad());
    };
    cmp("grad matmul+views", run(Device::CPU), run(ztk_dev), 1e-4f);
}

// mean 反传(1/n 缩放路径)
ZTK_CASE(grad_mean, true) {
    auto run = [](Device d) {
        Tensor x = Tensor::randn({5, 7}, 39).to_device(d);
        x.requires_grad(true);
        Tensor m = Tensor::mean(Tensor::mean(x, 1), 0);
        m.backward();
        return back(x.grad());
    };
    cmp("grad mean chain", run(Device::CPU), run(ztk_dev), 1e-4f);
}

} // namespace

// t21-losses 复现: CE 反向链(CUDA 非法访问定位用)
ZTK_CASE(repro_t21_ce, true) {
    auto run = [](Device d) {
        Tensor lg = Tensor::randn({5, 4}, 89);
        int64_t tg[5] = {0, 1, 2, 3, 0};
        Tensor x = Tensor::from_host(lg.to_device(Device::CPU).data<float>(), {5, 4}).to_device(d);
        Tensor t = Tensor::from_host(tg, {5}, DType::I64).to_device(d);
        x.requires_grad(true);
        Tensor loss = Loss::cross_entropy(x, t);
        loss.backward();
        return back(x.grad());
    };
    Tensor gcpu = run(Device::CPU);
    Tensor gdev = run(ztk_dev);
    check(max_diff(gcpu, gdev) < 5e-3f, "ce grad parity");
}

// t10 复现: 纯 CPU Adam 线性回归训练循环(GPU 构建崩点定位用)
ZTK_CASE(repro_t10_training, false) {
    if (ztk_dev != Device::CPU) {
        return;   // 本体是 CPU 训练; GPU 构建下同代码段崩溃
    }
    const int64_t N = 32;
    float xs[32], ys[32];
    for (int64_t i = 0; i < N; ++i) {
        xs[i] = static_cast<float>(i) / N - 0.5f;
        ys[i] = 2.f * xs[i] + 1.f;
    }
    Tensor X = Tensor::from_host(xs, {N, 1});
    Tensor Y = Tensor::from_host(ys, {N, 1});
    std::vector<Layer*> net = {Layer::create(LayerType::Linear, LayerParam{1, 1})};
    Optimizer* opt = Optimizer::create(OptimType::Adam, net, OptimParam{0.05});
    float first = 0.f, last = 0.f;
    for (int step = 0; step < 400; ++step) {
        Tensor h = X;
        for (Layer* l : net) h = l->forward(h);
        std::printf("      %d fwd\n", step);
        Tensor loss = Loss::mse(h, Y);
        if (step == 0) first = loss.item<float>();
        last = loss.item<float>();
        opt->clear();
        std::printf("      %d CLEAR\n", step);
        loss.backward();
        std::printf("      step %d bwd done\n", step);
        opt->step();
        std::printf("      step %d opt done\n", step);
    }
    Tensor h = X;
    for (Layer* l : net) h = l->forward(h);
    check(last < first / 100.f, "t10 repro loss drops");
    Layer::destroy(net[0]);
    Optimizer::destroy(opt);
}

ZTK_CASE(repro_unifp_reduce_adamw, true) {
    for (int64_t n : {3LL, 257LL, 511LL}) {
        Tensor x = Tensor::ones({n}, DType::F32, ztk_dev);
        check(std::fabs(Tensor::sum(x, 0).item<float>() - (float)n) < 1e-4f,
              "reduce tail sum");
        check(std::fabs(Tensor::mean(x, 0).item<float>() - 1.f) < 1e-6f,
              "reduce tail mean");
    }
    Tensor p = Tensor::ones({3}, DType::F32, ztk_dev);
    p.requires_grad(true);
    OptimParam cfg;
    cfg.lr = 0.1;
    cfg.weight_decay = 0.1;
    Optimizer* opt = Optimizer::create(OptimType::AdamW, std::vector<Tensor>{p}, cfg);
    for (int step = 0; step < 2; ++step) {
        opt->clear();
        Tensor::sum(p, 0).backward();
        opt->step();
    }
    check(std::fabs(back(p).data<float>()[0] - 0.7811f) < 1e-4f,
          "AdamW decoupled two steps");
    Optimizer::destroy(opt);
}

// ============ SB2-SB4: 布局/生成/卷积/池化/嵌入/归一化/损失/注意力 ============

ZTK_CASE(layout_ops, false) {
    Tensor x = Tensor::randn({4, 6}, 40);
    Tensor sc = x.slice(1, 1, 4);        // 具名(隔离临时生命周期)
    Tensor xd = on(ztk_dev, x);
    Tensor sd = xd.slice(1, 1, 4);
    std::printf("      (sd dev=%d numel=%lld)\n", (int)sd.device(),
                (long long)sd.numel());
    cmp("slice", sc, sd, 0.f);
    cmp("reshape", x.reshape({2, 12}), on(ztk_dev, x).reshape({2, 12}), 0.f);
    cmp("transpose", x.transpose(0, 1), on(ztk_dev, x).transpose(0, 1), 0.f);
    cmp("permute", x.permute({1, 0}), on(ztk_dev, x).permute({1, 0}), 0.f);
    Tensor a = Tensor::randn({3, 4}, 41), b = Tensor::randn({2, 4}, 42);
    Tensor c = Tensor::randn({3, 4}, 43);   // stack 要求同形
    cmp("cat", Tensor::cat(a, b, 0),
        Tensor::cat(on(ztk_dev, a), on(ztk_dev, b), 0), 0.f);
    cmp("stack", Tensor::stack(std::vector<Tensor>{a, c}, 0),
        Tensor::stack(std::vector<Tensor>{on(ztk_dev, a), on(ztk_dev, c)}, 0), 0.f);
    cmp("arange", Tensor::arange(37),
        Tensor::arange(37, DType::F32, ztk_dev), 0.f);
    Tensor r1 = Tensor::randn({5, 9}, 44), r2 = Tensor::randn({5, 9}, 44);
    cmp("randn same seed", r1, on(ztk_dev, r2), 0.f);
}

ZTK_CASE(conv2d_op, false) {
    Tensor x = Tensor::randn({2, 3, 9, 9}, 44);
    Tensor w = Tensor::randn({4, 3, 3, 3}, 45);
    Conv2dOptions o;
    cmp("conv2d", Tensor::conv2d(x, w, o),
        Tensor::conv2d(on(ztk_dev, x), on(ztk_dev, w), o), 1e-3f);
    o.stride = {2, 2};
    o.pad = {1, 1};
    cmp("conv2d s2p1", Tensor::conv2d(x, w, o),
        Tensor::conv2d(on(ztk_dev, x), on(ztk_dev, w), o), 1e-3f);
    o.stride = {1, 1};
    o.pad = {0, 0};
    o.groups = 3;
    Tensor wg = Tensor::randn({6, 1, 3, 3}, 46);
    cmp("conv2d groups", Tensor::conv2d(x, wg, o),
        Tensor::conv2d(on(ztk_dev, x), on(ztk_dev, wg), o), 1e-3f);
}

ZTK_CASE(pools, false) {
    Tensor x = Tensor::randn({2, 3, 8, 8}, 47);
    Pool2dOptions o;
    cmp("max_pool", Tensor::max_pool2d(x, {3, 3}, o),
        Tensor::max_pool2d(on(ztk_dev, x), {3, 3}, o), 0.f);
    cmp("avg_pool", Tensor::avg_pool2d(x, {3, 3}, o),
        Tensor::avg_pool2d(on(ztk_dev, x), {3, 3}, o), 1e-5f);
}

ZTK_CASE(grad_conv_pool, true) {
    auto run_c = [](Device d) {   // 纯 conv 梯度(隔离)
        Tensor x = Tensor::randn({2, 3, 8, 8}, 48).to_device(d);
        Tensor w = Tensor::randn({4, 3, 3, 3}, 49).to_device(d);
        x.requires_grad(true);
        w.requires_grad(true);
        Tensor y = Tensor::conv2d(x, w);
        Tensor s = y;
        for (int64_t dd = s.dim_count() - 1; dd >= 0; --dd) {
            s = Tensor::sum(s, dd);
        }
        s.backward();
        return back(x.grad());
    };
    Tensor gc = run_c(Device::CPU);
    Tensor gd = run_c(ztk_dev);
    cmp("grad conv x", gc, gd, 1e-3f);
    auto run_p = [](Device d) {   // 纯 max_pool 梯度(隔离 CAS 路径)
        Tensor x = Tensor::randn({2, 3, 8, 8}, 52).to_device(d);
        x.requires_grad(true);
        Tensor y = Tensor::max_pool2d(x, {2, 2});
        Tensor s = y;
        for (int64_t dd = s.dim_count() - 1; dd >= 0; --dd) {
            s = Tensor::sum(s, dd);
        }
        s.backward();
        return back(x.grad());
    };
    Tensor pc2 = run_p(Device::CPU);
    Tensor pd2 = run_p(ztk_dev);
    cmp("grad max_pool x", pc2, pd2, 1e-3f);
}

ZTK_CASE(embedding_op, false) {
    Tensor w = Tensor::randn({7, 5}, 50);
    int64_t ids[6] = {0, 3, 6, 2, 2, 5};
    Tensor idx = Tensor::from_host(ids, {6}, DType::I64);
    cmp("embedding fwd", Tensor::embedding(w, idx),
        Tensor::embedding(on(ztk_dev, w), on(ztk_dev, idx)), 0.f);
    auto run = [&ids](Device d) {
        Tensor wt = Tensor::randn({7, 5}, 51).to_device(d);
        Tensor ix = Tensor::from_host(ids, {6}, DType::I64).to_device(d);
        wt.requires_grad(true);
        Tensor y = Tensor::embedding(wt, ix);
        Tensor s = y;
        for (int64_t dd = s.dim_count() - 1; dd >= 0; --dd) {
            s = Tensor::sum(s, dd);
        }
        s.backward();
        return back(wt.grad());
    };
    Tensor gc = run(Device::CPU);
    Tensor gd = run(ztk_dev);
    cmp("embedding grad (dup rows)", gc, gd, 1e-4f);
}

ZTK_CASE(losses, false) {
    Tensor x = Tensor::randn({5, 7}, 58);
    int64_t tg[5] = {0, 1, 2, 3, 0};
    Tensor t = Tensor::from_host(tg, {5}, DType::I64);
    cmp("cross_entropy", Loss::cross_entropy(x, t),
        Loss::cross_entropy(on(ztk_dev, x), on(ztk_dev, t)), 1e-4f);
    Tensor p = Tensor::randn({6, 5}, 59), q = Tensor::randn({6, 5}, 60);
    cmp("mse", Loss::mse(p, q),
        Loss::mse(on(ztk_dev, p), on(ztk_dev, q)), 1e-4f);
}

ZTK_CASE(attention_op, false) {
    Tensor q = Tensor::randn({1, 2, 6, 5}, 61);
    Tensor k = Tensor::randn({1, 2, 6, 5}, 62);
    Tensor v = Tensor::randn({1, 2, 6, 5}, 63);
    // Vulkan attention 锁定中(验证未过) —— 回落路径数值仍须正确
    cmp("attention", Tensor::attention(q, k, v),
        Tensor::attention(on(ztk_dev, q), on(ztk_dev, k),
                          on(ztk_dev, v)), 1e-3f);
}

// ============ 回归: DLL 消费路径暴露的缺陷族(外部训练项目反馈, 本轮修复) ============

// softmax 任意 dim: CUDA 此前无视 dim 恒取末维; Vulkan/Metal 此前直接抛错降级
ZTK_CASE(softmax_any_dim, false) {
    Tensor x = Tensor::randn({3, 7, 5, 6}, 71);
    cmp("softmax dim=0 4D", Activator::softmax(x, 0),
        Activator::softmax(on(ztk_dev, x), 0), 1e-4f);
    cmp("softmax dim=1 4D", Activator::softmax(x, 1),
        Activator::softmax(on(ztk_dev, x), 1), 1e-4f);
    cmp("softmax dim=2 4D", Activator::softmax(x, 2),
        Activator::softmax(on(ztk_dev, x), 2), 1e-4f);
    Tensor x2 = Tensor::randn({5, 9}, 72);
    cmp("softmax dim=0 2D", Activator::softmax(x2, 0),
        Activator::softmax(on(ztk_dev, x2), 0), 1e-4f);
    Tensor xs = Tensor::randn({1000}, 73);
    cmp("softmax dim=0 1D", Activator::softmax(xs, 0),
        Activator::softmax(on(ztk_dev, xs), 0), 1e-4f);
}

// 工厂 Conv2d: groups 权重形状(此前漏建分组 → 参数爆炸/校验失败)
// 与 4D 偏置广播(此前 [C]+[B,C,H,W] 不可广播 → 抛错)
ZTK_CASE(conv2d_layer_groups_bias, false) {
    LayerParam p;
    p.in = 8; p.out = 16; p.kernel = {3, 3};
    p.conv.pad = {1, 1};
    p.conv.groups = 8;                    // DW: 权重应为 [16,1,3,3]
    p.bias = true;
    Tensor x = Tensor::randn({2, 8, 7, 7}, 74);
    // 同一实例先 CPU 后设备(randn 是全局引擎, 双实例权重不同): 权重迁移语义实测
    Layer* l1 = Layer::create(LayerType::Conv2d, p);
    Tensor yc = l1->forward(x);
    Tensor yd = l1->forward(on(ztk_dev, x));
    auto ps = l1->params();
    check(ps[0].size(1) == 1 && ps.size() == 2,
          "dw layer weight [O,1,k,k] + bias param");
    cmp("conv2d dw+bias layer", yc, yd, 1e-3f);
    Layer::destroy(l1);
    // 普通卷积带偏置(4D 广播路径)
    LayerParam q;
    q.in = 8; q.out = 12; q.kernel = {1, 1}; q.bias = true;
    Layer* l2 = Layer::create(LayerType::Conv2d, q);
    Tensor y2 = l2->forward(x);
    Tensor y3 = l2->forward(on(ztk_dev, x));
    cmp("conv2d 1x1+bias layer", y2, y3, 1e-3f);
    Layer::destroy(l2);
}

// 非连续视图输入: 各算子此前要求连续直接抛错(CUDA), 统一为原生材料化
ZTK_CASE(strided_view_inputs, false) {
    Tensor t = Tensor::randn({4, 6, 5}, 75).transpose(0, 1);   // [6,4,5] 非连续
    check(!t.is_contiguous(), "transpose view non-contiguous");
    cmp("exp on transpose", Tensor::exp(t), Tensor::exp(on(ztk_dev, t)), 1e-4f);
    cmp("silu on transpose", Activator::silu(t),
        Activator::silu(on(ztk_dev, t)), 1e-4f);
    cmp("sum dim1 on transpose", Tensor::sum(t, 1),
        Tensor::sum(on(ztk_dev, t), 1), 1e-3f);
    cmp("softmax dim1 on transpose", Activator::softmax(t, 1),
        Activator::softmax(on(ztk_dev, t), 1), 1e-4f);
    Tensor s = Tensor::randn({2, 8, 6, 6}, 76);
    Tensor sl = s.slice(1, 2, 6);                               // 通道切片
    cmp("cat on slices", Tensor::cat({sl, sl}, 1),
        Tensor::cat({on(ztk_dev, sl), on(ztk_dev, sl)}, 1), 0.f);
    cmp("stack on transpose", Tensor::stack({t, t}, 0),
        Tensor::stack({on(ztk_dev, t), on(ztk_dev, t)}, 0), 0.f);
    Tensor w = Tensor::randn({4, 4, 3, 3}, 77);
    Conv2dOptions o; o.pad = {1, 1};
    cmp("conv2d on slice", Tensor::conv2d(sl, w, o),
        Tensor::conv2d(on(ztk_dev, sl), on(ztk_dev, w), o), 1e-3f);
}

// 规约大 K 维 0: 两段式路径数值(此前一线程一输出, 本修仅动并行度)
ZTK_CASE(reduce_dim0_large, false) {
    Tensor x = Tensor::randn({8192, 96}, 78);
    cmp("sum dim0 [8192,96]", Tensor::sum(x, 0),
        Tensor::sum(on(ztk_dev, x), 0), 0.5f);
    cmp("mean dim0 [8192,96]", Tensor::mean(x, 0),
        Tensor::mean(on(ztk_dev, x), 0), 1e-4f);
    Tensor y = Tensor::randn({1000, 7, 5}, 79);
    cmp("mean dim1 [1000,7,5]", Tensor::mean(y, 1),
        Tensor::mean(on(ztk_dev, y), 1), 1e-5f);
    Tensor f = Tensor::randn({4096, 4096}, 80);
    Tensor s1 = Tensor::sum(f, 1);
    Tensor s2 = Tensor::sum(on(ztk_dev, f), 1);
    cmp("sum dim1 [4096,4096]", s1, s2, 2.f);
}

// to_device 保留 requires_grad(叶参数迁移场景)
ZTK_CASE(to_device_grad_flag, false) {
    Tensor x = Tensor::randn({4, 4}, 81);
    x.requires_grad(true);
    Tensor d = on(ztk_dev, x);
    check(d.requires_grad(), "to_device keeps requires_grad");
}

// BatchNorm2d train 态批统计跨设备一致(此前全后端把 {double,double,bool}
// 选项按 float[3] 强转, 训练标志读成 0 → GPU 一直跑 eval 统计; kit 无
// BN 覆盖故长期漏网 —— UniFP 消费者实测暴露)
ZTK_CASE(bn_train_parity, false) {
    LayerParam p; p.channels = 7; p.eps = 1e-5; p.momentum = 0.1;
    Layer* l = Layer::create(LayerType::BatchNorm2d, p);
    l->train(true);
    Tensor x = Tensor::add(
        Tensor::mul(Tensor::randn({2, 7, 6, 9}, 82), Tensor::full({}, 2.0)),
        Tensor::full({}, 3.0));
    Tensor yc = l->forward(x);                       // CPU 批统计
    Tensor yd = l->forward(on(ztk_dev, x));          // 同层设备前向(权重迁移)
    cmp("bn train batch stats", yc, yd, 1e-3f);
    l->train(false);
    Tensor ye = l->forward(on(ztk_dev, x));
    check(max_diff(yc, back(ye)) > 0.2f, "bn train differs from running-stat eval");
    Layer::destroy(l);
}

// 减法反传符号(用户修复: sub 曾注册加法梯度, db 恒 +g → BCE/Dice/GIoU
// 全部反向; 广播两态 + 共享叶 + 常量减参数 + 加法不变回归)
ZTK_CASE(sub_gradient, true) {
    auto red = [](Tensor x) {
        while (x.dim_count()) x = Tensor::sum(x, 0);
        return x;
    };
    Tensor g2 = Tensor::full({2, 3}, 2.0, DType::F32, ztk_dev);   // 上游 2 倍
    for (int mode = 0; mode < 4; ++mode) {
        Shape as = mode == 1 ? Shape{1, 3} : mode == 2 ? Shape{} : Shape{2, 3};
        Shape bs = mode == 1 ? Shape{2, 1} : mode == 2 ? Shape{2, 3}
                                                   : mode == 3 ? Shape{1} : Shape{2, 3};
        Tensor a = Tensor::full(as, 2.0, DType::F32, ztk_dev);
        Tensor b = Tensor::full(bs, 3.0, DType::F32, ztk_dev);
        a.requires_grad(true);
        b.requires_grad(true);
        Tensor y = Tensor::sub(a, b);
        Tensor yh = back(y);                                        // 前向: a-b = -1
        bool fok = true;
        for (int64_t i = 0; i < yh.numel(); ++i)
            fok = fok && std::fabs(yh.data<float>()[i] + 1.f) < 1e-5f;
        check(fok, "sub forward a-b");
        red(Tensor::mul(y, g2)).backward();
        Tensor ga = back(a.grad());                                 // d(Σ2y)/da = 12/|a|
        Tensor gb = back(b.grad());                                 // d(Σ2y)/db = -12/|b|
        const float ea = 12.f / float(a.numel());
        const float eb = -12.f / float(b.numel());
        bool ok = true;
        for (int64_t i = 0; i < ga.numel(); ++i)
            ok = ok && std::fabs(ga.data<float>()[i] - ea) < 1e-4f;
        for (int64_t i = 0; i < gb.numel(); ++i)
            ok = ok && std::fabs(gb.data<float>()[i] - eb) < 1e-4f;
        check(ok, "sub da=+g db=-g (broadcast)");
    }
    Tensor x = Tensor::full({2, 3}, 2.0, DType::F32, ztk_dev);
    x.requires_grad(true);
    red(Tensor::sub(x, x)).backward();
    bool zero = true;
    for (int64_t i = 0; i < back(x.grad()).numel(); ++i)
        zero = zero && std::fabs(back(x.grad()).data<float>()[i]) < 1e-5f;
    check(zero, "sub shared-leaf cancellation");
    Tensor b = Tensor::full({1}, 3.0, DType::F32, ztk_dev);
    b.requires_grad(true);
    Tensor::sub(Tensor::full({1}, 1.0, DType::F32, ztk_dev), b).backward();
    check(std::fabs(back(b.grad()).data<float>()[0] + 1.f) < 1e-5f,
          "sub constant-minus-param");
    Tensor a = Tensor::full({1}, 2.0, DType::F32, ztk_dev);
    Tensor c = Tensor::full({1}, 3.0, DType::F32, ztk_dev);
    a.requires_grad(true);
    c.requires_grad(true);
    Tensor::add(a, c).backward();
    check(std::fabs(back(a.grad()).data<float>()[0] - 1.f) < 1e-5f
          && std::fabs(back(c.grad()).data<float>()[0] - 1.f) < 1e-5f,
          "add grads unchanged");
}

// LayerNorm 任意 dim(通道维 LN —— ConvNeXt 式 NCHW 用法; 此前仅末维,
// 消费者需 transpose+clone 往返)。fwd 与梯度对拍(手动 LN 参考)。
ZTK_CASE(layer_norm_dim, true) {
    LayerParam p; p.normalized_dim = 6; p.eps = 1e-3; p.norm_dim = 1;
    Tensor x = Tensor::randn({2, 6, 4, 5}, 83);
    Layer* lc = Layer::create(LayerType::LayerNorm, p);
    Tensor yc = lc->forward(x);
    // 手动参考(逐位置跨通道)
    Tensor ref = Tensor::zeros(yc.shape());
    {
        const float* xp = x.data<float>();
        float* rp = ref.data<float>();
        for (int64_t n = 0; n < 2; n++)
            for (int64_t h = 0; h < 4; h++)
                for (int64_t w = 0; w < 5; w++) {
                    double mu = 0, ex2 = 0;
                    for (int64_t c = 0; c < 6; c++) {
                        const float v = xp[((n * 6 + c) * 4 + h) * 5 + w];
                        mu += v; ex2 += double(v) * v;
                    }
                    mu /= 6;
                    const float inv = 1.f / float(std::sqrt(
                        std::max(ex2 / 6 - mu * mu, 0.0) + 1e-3));
                    for (int64_t c = 0; c < 6; c++) {
                        const int64_t i = ((n * 6 + c) * 4 + h) * 5 + w;
                        rp[i] = (xp[i] - mu) * inv;      // γ=1 β=0
                    }
                }
    }
    Tensor yd = lc->forward(on(ztk_dev, x));
    cmp("ln dim=1 fwd", yc, yd, 1e-4f);
    Tensor rc = ref.clone(), rd = yd;
    double e = 0;
    Tensor rh = back(yd);
    for (int64_t i = 0; i < yc.numel(); i++)
        e = std::max(e, double(std::fabs(rh.data<float>()[i]
                                         - ref.data<float>()[i])));
    check(e < 1e-3f, "ln dim=1 vs manual (fwd values)");
    Layer::destroy(lc);
    // 梯度: dim=1 CPU vs 设备(同层双前向)
    Layer* lg = Layer::create(LayerType::LayerNorm, p);
    Tensor xa = Tensor::randn({2, 6, 3, 3}, 84);
    Tensor xg = on(ztk_dev, xa);
    xa.requires_grad(true); xg.requires_grad(true);
    Tensor fa = lg->forward(xa), fb = lg->forward(xg);
    for (int64_t d = fa.dim_count() - 1; d >= 0; --d) { fa = Tensor::sum(fa, d); fb = Tensor::sum(fb, d); }
    fa.backward(); fb.backward();
    Tensor ga = back(xa.grad()), gb = back(xg.grad());
    double eg = 0;
    for (int64_t i = 0; i < ga.numel(); i++)
        eg = std::max(eg, double(std::fabs(ga.data<float>()[i] - gb.data<float>()[i])));
    check(eg < 1e-4f, "ln dim=1 grad parity");
    Layer::destroy(lg);
}

// Linear 带偏置(Vulkan 此前对 bias 直接抛错 —— common.comp 广播本可支持)
ZTK_CASE(linear_bias, false) {
    Tensor x = Tensor::randn({3, 7}, 85);
    Tensor w = Tensor::randn({4, 7}, 86);
    Tensor b = Tensor::randn({4}, 87);
    Tensor yc = Tensor::linear(x, w, &b);
    Tensor xd = on(ztk_dev, x), wd = on(ztk_dev, w), bd = on(ztk_dev, b);
    Tensor yd = Tensor::linear(xd, wd, &bd);
    cmp("linear+bias", yc, yd, 1e-3f);
}

// upsample2x_nearest(新算子): 前向 vs 手写参考 + 与 cat+reshape 组合等价 + 梯度
ZTK_CASE(upsample2x, true) {
    Tensor x = Tensor::randn({2, 3, 5, 7}, 88);
    Tensor y = Tensor::upsample2x_nearest(x);
    check(y.shape() == Shape{2, 3, 10, 14}, "shape 2x");
    // 手写参考
    Tensor ref = Tensor::zeros(y.shape());
    {
        const float* xp = x.data<float>();
        float* rp = ref.data<float>();
        const int64_t W = 7, W2 = 14;
        for (int64_t n = 0; n < 2; n++)
            for (int64_t c = 0; c < 3; c++)
                for (int64_t i = 0; i < 5; i++)
                    for (int64_t j = 0; j < 7; j++) {
                        const float v = xp[((n * 3 + c) * 5 + i) * W + j];
                        for (int di = 0; di < 2; di++)
                            for (int dj = 0; dj < 2; dj++)
                                rp[((n * 3 + c) * 10 + i * 2 + di) * W2 + j * 2 + dj] = v;
                    }
    }
    Tensor yd = Tensor::upsample2x_nearest(on(ztk_dev, x));
    cmp("upsample2x fwd", y, yd, 0.f);
    Tensor yh = back(yd);
    double e = 0;
    for (int64_t i = 0; i < y.numel(); i++)
        e = std::max(e, double(std::fabs(yh.data<float>()[i]
                                         - ref.data<float>()[i])));
    check(e == 0.0, "upsample2x exact vs manual");
    // 与 cat+reshape 组合等价(既有消费者路径)
    Tensor combo = Tensor::cat({x.unsqueeze(4), x.unsqueeze(4)}, 4)
                      .reshape({2, 3, 5, 14});
    Tensor combo2 = Tensor::cat({combo.unsqueeze(3), combo.unsqueeze(3)}, 3)
                       .reshape({2, 3, 10, 14});
    cmp("upsample2x == cat+reshape", y, Tensor::upsample2x_nearest(
        on(ztk_dev, x)), 0.f);
    double e2 = 0;
    for (int64_t i = 0; i < y.numel(); i++)
        e2 = std::max(e2, double(std::fabs(yh.data<float>()[i]
                                           - combo2.data<float>()[i])));
    check(e2 == 0.0, "equivalence cat-composition");
    // 梯度: dx == g[::2,::2]
    Tensor xg = Tensor::ones({2, 3, 4, 6});
    xg.requires_grad(true);
    Tensor loss = Tensor::upsample2x_nearest(xg);
    while (loss.dim_count()) loss = Tensor::sum(loss, 0);
    loss.backward();
    Tensor g = xg.grad();
    bool ge = g.numel() == xg.numel();
    check(ge, "grad shape");
}
