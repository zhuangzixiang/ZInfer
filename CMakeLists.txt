cmake_minimum_required(VERSION 3.16)
project(zinfer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 源码为 UTF-8(含中文注释), MSVC 默认按本地代码页解析会破坏语法
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

option(ZNN_BUILD_TESTS    "Build tests"     ON)
option(ZNN_BUILD_EXAMPLES "Build examples"  ON)
# 里程碑 2+ 启用的后端开关: ZNN_CUDA / ZNN_METAL / ZNN_VULKAN / ZNN_ONEDNN
# 唯一默认三方依赖 Eigen 于实现阶段引入 third_party/eigen

# 算子源清单: 新增算子在此加一行(与 operator_list.cpp 的注册行对应)
set(ZNN_OPERATORS
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
    src/operators/linear.cpp
    src/operators/cat.cpp
    src/operators/stack.cpp
    src/operators/conv2d.cpp
    src/operators/max_pool2d.cpp
    src/operators/avg_pool2d.cpp
    src/operators/embedding.cpp
    src/operators/attention.cpp
    src/operators/cross_entropy.cpp
    src/operators/mse.cpp
    src/operators/operator_list.cpp
)

add_library(znn
    src/znn_internal.cpp
    ${ZNN_OPERATORS}
)
target_include_directories(znn PUBLIC include PRIVATE src)

if(ZNN_BUILD_TESTS)
    enable_testing()
    add_executable(znn_test tests/test_smoke.cpp)
    target_link_libraries(znn_test PRIVATE znn)
    add_test(NAME smoke COMMAND znn_test)
endif()

if(ZNN_BUILD_EXAMPLES)
    add_subdirectory(examples/mnist)
endif()
