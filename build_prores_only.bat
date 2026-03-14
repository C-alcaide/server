@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;%PATH%
"C:\Program Files\CMake\bin\cmake.exe" --build "d:\Github\CasparCG-cuda\out\build\x64-RelWithDebInfo" --target cuda_prores -- -j8
