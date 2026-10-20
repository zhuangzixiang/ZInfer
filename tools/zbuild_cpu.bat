@echo off
rem zbuild_cpu.bat — CPU 构建(VS; 铁律: 先删旧二进制再构建, 构建失败即止)
set MSVC=E:\Tools\IDEs\VS\2019\Community\VC\Tools\MSVC\14.29.30133
set WININC=E:\Windows Kits\10\Include\10.0.22621.0
set WINLIB=E:\Windows Kits\10\Lib\10.0.22621.0
set PATH=%MSVC%\bin\Hostx64\x64;E:\Tools\IDEs\Cmake\cmake-3.22.1\bin;%PATH%
set INCLUDE=%MSVC%\include;%WININC%\ucrt;%WININC%\um;%WININC%\shared
set LIB=%MSVC%\lib\x64;%WINLIB%\ucrt\x64;%WINLIB%\um\x64
cd /d E:\Documents\ZShared\ZProjs\ZSpeedAI\ZInfer
del /q build\Release\znn_test.exe build\Release\znn_bench.exe build\ztestkit\Release\ztestkit.exe build\examples\mnist\Release\mnist.exe 2>nul
cmake --build build --config Release || exit /b 1
