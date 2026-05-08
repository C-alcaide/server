@echo off
call "C:\Program Files\Microsoft Visual Studio\2026\Community\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
"C:\Program Files\CMake\bin\cmake.exe" --build d:\Github\CasparVP\build --target casparcg