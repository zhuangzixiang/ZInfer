# CMake generated Testfile for 
# Source directory: E:/Documents/ZShared/ZProjs/ZSpeedAI/ZInfer
# Build directory: E:/Documents/ZShared/ZProjs/ZSpeedAI/ZInfer/build-cuda
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(smoke "E:/Documents/ZShared/ZProjs/ZSpeedAI/ZInfer/build-cuda/znn_test.exe")
set_tests_properties(smoke PROPERTIES  _BACKTRACE_TRIPLES "E:/Documents/ZShared/ZProjs/ZSpeedAI/ZInfer/CMakeLists.txt;132;add_test;E:/Documents/ZShared/ZProjs/ZSpeedAI/ZInfer/CMakeLists.txt;0;")
subdirs("examples/mnist")
