@echo off
rem zbuild_cuda.bat — CUDA 构建(铁律: 先删旧二进制再构建, 构建失败即止)
set MSVC=E:\Tools\IDEs\VS\2019\Community\VC\Tools\MSVC\14.29.30133
set WININC=E:\Windows Kits\10\Include\10.0.22621.0
set WINLIB=E:\Windows Kits\10\Lib\10.0.22621.0
set CUDAROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4
set PATH=%MSVC%\bin\Hostx64\x64;E:\Windows Kits\10\bin\10.0.22621.0\x64;E:\Tools\IDEs\Cmake\cmake-3.22.1\bin;%CUDAROOT%\bin;%PATH%
set INCLUDE=%MSVC%\include;%WININC%\ucrt;%WININC%\um;%WININC%\shared;%CUDAROOT%\include
set LIB=%MSVC%\lib\x64;%WINLIB%\ucrt\x64;%WINLIB%\um\x64;%CUDAROOT%\lib\x64
set NVCC_PREPEND_FLAGS=-ccbin %MSVC%\bin\Hostx64\x64
cd /d E:\Documents\ZShared\ZProjs\ZSpeedAI\ZInfer
del /q build-cuda\ztestkit\ztestkit.exe build-cuda\znn_test.exe build-cuda\znn_bench.exe build-cuda\examples\mnist\mnist.exe 2>nul
cmake --build build-cuda || exit /b 1
