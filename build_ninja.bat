@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;%PATH%
"C:\Program Files\CMake\bin\cmake.exe" ^
    -B "d:\Github\CasparVP\build" ^
    -S "d:\Github\CasparVP\src" ^
    -G Ninja ^
    -DCMAKE_MAKE_PROGRAM="C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CUDA_ARCHITECTURES="52;61;80;86;89" ^
    "-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler -Wno-deprecated-gpu-targets" ^
    "-DCMAKE_CUDA_HOST_COMPILER=C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.50.35717/bin/Hostx64/x64/cl.exe"

"C:\Program Files\CMake\bin\cmake.exe" --build "d:\Github\CasparVP\build" --target casparcg
"C:\Program Files\CMake\bin\cmake.exe" --build "d:\Github\CasparVP\build" --target casparcg_copy_dependencies
