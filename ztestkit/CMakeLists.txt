# ztestkit — 独立测试工具包(可剥离: 仅依赖被测库公共头)
add_executable(ztestkit ztestkit.cpp cases/matrix.cpp)
target_link_libraries(ztestkit PRIVATE znn)
target_compile_options(ztestkit PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/Zi>)
target_link_options(ztestkit PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/DEBUG>)

add_executable(zbench bench.cpp)
target_link_libraries(zbench PRIVATE znn)

# DLL 外部消费者冒烟(仅公共头; DLL 模式下验证导出完整性)
add_executable(hello_dll ../examples/hello_dll/main.cpp)
target_link_libraries(hello_dll PRIVATE znn)
target_compile_options(hello_dll PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
