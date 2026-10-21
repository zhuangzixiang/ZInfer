@echo off
rem zbuild_dll.bat - Three-backend DLL (CUDA+Vulkan+CPU, delete-first)
set MSVC=E:\Tools\IDEs\VS\2019\Community\VC\Tools\MSVC\14.29.30133
set WININC=E:\Windows Kits\10\Include\10.0.22621.0
set WINLIB=E:\Windows Kits\10\Lib\10.0.22621.0
set CUDAROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4
set PATH=%MSVC%\bin\Hostx64\x64;E:\Windows Kits\10\bin\10.0.22621.0\x64;E:\Tools\IDEs\Cmake\cmake-3.22.1\bin;%CUDAROOT%\bin;%PATH%
set INCLUDE=%MSVC%\include;%WININC%\ucrt;%WININC%\um;%WININC%\shared;%CUDAROOT%\include
set LIB=%MSVC%\lib\x64;%WINLIB%\ucrt\x64;%WINLIB%\um\x64;%CUDAROOT%\lib\x64
set NVCC_PREPEND_FLAGS=-ccbin %MSVC%\bin\Hostx64\x64
cd /d E:\Documents\ZShared\ZProjs\ZSpeedAI\ZInfer

del /q build-dll\znn.dll build-dll\znn.lib 2>nul

if not exist build-dll\CMakeCache.txt (
    cmake -S . -B build-dll -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="/utf-8" -DZNN_CUDA=ON -DZNN_VULKAN=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DZNN_BUILD_DLL=ON -DZNN_BUILD_TESTS=OFF -DZNN_BUILD_TESTKIT=OFF || exit /b 1
)
cmake --build build-dll --target znn || exit /b 1
