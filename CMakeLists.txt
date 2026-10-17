cmake_minimum_required(VERSION 3.16)
project(zinfer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 源码为 UTF-8(含中文注释), MSVC 默认按本地代码页解析会破坏语法。
# CXX 直接用 /utf-8; CUDA 经 nvcc 的 -Xcompiler 转发给主机编译器 cl.exe。
add_compile_options("$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CXX>>:/utf-8>")
if(MSVC AND ZNN_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-utf-8")
endif()

# Debug 默认带 /RTC1, 与 Eigen 内核的 /O2 冲突; 本项目不依赖运行时检查
if(MSVC)
    string(REPLACE "/RTC1" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
endif()

option(ZNN_BUILD_TESTS    "Build tests"     ON)
option(ZNN_BUILD_EXAMPLES "Build examples"  ON)
option(ZNN_CUDA "Build CUDA backend (requires nvcc)" OFF)
# 里程碑 2+ 启用的后端开关: ZNN_CUDA / ZNN_METAL / ZNN_VULKAN / ZNN_ONEDNN
# 唯一默认三方依赖 Eigen 于实现阶段引入 third_party/eigen

# 算子源清单: 新增算子在此加一行(与 operator_list.cpp 的注册行对应);
# 异构后端内核为同名异后缀文件(conv2d.cu/.mm/.comp), 里程碑 2+ 按开关加入
set(ZNN_OPERATORS
    src/operators/shared.cpp
    src/operators/arange.cpp
    src/operators/randn.cpp
    src/operators/add.cpp
    src/operators/sub.cpp
    src/operators/mul.cpp
    src/operators/div.cpp
    src/operators/neg.cpp
    src/operators/exp.cpp
    src/operators/log.cpp
    src/operators/sqrt.cpp
    src/operators/relu.cpp
    src/operators/gelu.cpp
    src/operators/sigmoid.cpp
    src/operators/tanh.cpp
    src/operators/silu.cpp
    src/operators/softmax.cpp
    src/operators/sum.cpp
    src/operators/mean.cpp
    src/operators/matmul.cpp
    src/operators/matmul_eigen.cpp
    src/operators/linear_eigen.cpp
    src/operators/conv2d_eigen.cpp
    src/operators/linear.cpp
    src/operators/cat.cpp
    src/operators/stack.cpp
    src/operators/conv2d.cpp
    src/operators/max_pool2d.cpp
    src/operators/avg_pool2d.cpp
    src/operators/layer_norm.cpp
    src/operators/batch_norm2d.cpp
    src/operators/attention.cpp
    src/operators/embedding.cpp
    src/operators/cross_entropy.cpp
    src/operators/mse.cpp
    src/operators/transpose.cpp
    src/operators/reshape.cpp
    src/operators/permute.cpp
    src/operators/slice.cpp
    src/operators/operator_list.cpp
)

# 层源清单: 一层一文件(ActivationLayer 服务 4 种激活, Pool2dLayer 服务 2 种池化)
set(ZNN_LAYERS
    src/layers/layer_factory.cpp
    src/layers/linear.cpp
    src/layers/conv2d.cpp
    src/layers/batch_norm2d.cpp
    src/layers/layer_norm.cpp
    src/layers/embedding.cpp
    src/layers/dropout.cpp
    src/layers/activation.cpp
    src/layers/flatten.cpp
    src/layers/pool2d.cpp
)

# 优化器源清单: 一个优化器一文件
set(ZNN_OPTIM
    src/optim/optimizer_factory.cpp
    src/optim/sgd.cpp
    src/optim/adam.cpp
    src/optim/adamw.cpp
)

if(ZNN_CUDA)
    enable_language(CUDA)
    # dynlib 包含 cublasLt.h(CUDA 头文件), 必须用 nvcc 编译
    set_source_files_properties(src/dynlib.cpp PROPERTIES LANGUAGE CUDA)
    set(ZNN_CUDA_SRC
        src/backends/cuda/cuda_device.cu
        src/backends/cuda/cuda_gemm.cu
        src/backends/cuda/cuda_kernels.cu
        src/backends/cuda/cuda_kernels2.cu
        src/backends/cuda/cuda_kernels3.cu
        src/backends/cuda/linear_cuda.cu)
    set(CUDA_STANDARD 17)
else()
    set(ZNN_CUDA_SRC src/backends/cuda/cuda_stub.cpp)
endif()

add_library(znn
    src/znn_internal.cpp
    src/grad_scaler.cpp
    src/dynlib.cpp
    src/autograd.cpp
    src/op_dispatcher.cpp
    ${ZNN_OPERATORS}
    ${ZNN_LAYERS}
    ${ZNN_OPTIM}
    ${ZNN_CUDA_SRC}
)
target_include_directories(znn PUBLIC include PRIVATE src third_party/eigen)
if(ZNN_CUDA)
    target_compile_definitions(znn PRIVATE ZNN_CUDA)
endif()

# Eigen 多线程 GEMM 走 OpenMP(MSVC /openmp; gcc/clang -fopenmp 由下方统一)
if(MSVC AND NOT ZNN_CUDA)
    target_compile_options(znn PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/openmp>")
else()
    find_package(OpenMP REQUIRED)
    target_link_libraries(znn PRIVATE OpenMP::OpenMP_CXX)
endif()

# Eigen: 唯一三方依赖(header-only), kernels[CPU] 快速内核使用

if(ZNN_BUILD_TESTS)
    enable_testing()
    add_executable(znn_test tests/test_smoke.cpp)
    target_include_directories(znn_test PRIVATE src)
    target_link_libraries(znn_test PRIVATE znn)
    add_test(NAME smoke COMMAND znn_test)

    add_executable(znn_bench tests/bench.cpp)
    target_link_libraries(znn_bench PRIVATE znn)

    if(ZNN_CUDA)
        add_executable(gpu_smoke tests/gpu_smoke.cpp)
        target_link_libraries(gpu_smoke PRIVATE znn)
        add_executable(gpu_bench tests/gpu_bench.cpp)
        target_link_libraries(gpu_bench PRIVATE znn)
        set_source_files_properties(tests/gpu_bench.cpp PROPERTIES LANGUAGE CUDA)
        add_executable(hello tests/hello.cpp)
        target_link_libraries(hello PRIVATE znn)
    endif()
endif()

if(ZNN_BUILD_EXAMPLES)
    add_subdirectory(examples/mnist)
endif()
