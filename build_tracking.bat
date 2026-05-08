@echo off
call "C:\Program Files\Microsoft Visual Studio\2026\Community\VC\Auxiliary\Build\vcvars64.bat"
"C:\Program Files\CMake\bin\cmake.exe" --build "d:\Github\CasparVP\build" --target tracking
