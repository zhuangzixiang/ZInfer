# ztestkit — 独立测试工具包(可剥离: 仅依赖被测库公共头)
add_executable(ztestkit ztestkit.cpp cases/matrix.cpp)
target_link_libraries(ztestkit PRIVATE znn)
