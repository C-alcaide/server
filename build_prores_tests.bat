@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

cl 2>&1 | findstr /i "version"
echo.

set BUILD_DIR=d:\Github\CasparCG-cuda\tests_standalone\build
set SRC_DIR=d:\Github\CasparCG-cuda\tests_standalone

echo === Configuring ===
cmake -S "%SRC_DIR%" -B "%BUILD_DIR%" ^
  -G "Ninja" ^
  -DCMAKE_BUILD_TYPE=RelWithDebInfo ^
  -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler -Wno-deprecated-gpu-targets" ^
  -DCMAKE_C_COMPILER=cl ^
  -DCMAKE_CXX_COMPILER=cl

if %ERRORLEVEL% neq 0 (
  echo === Configure FAILED ===
  exit /b %ERRORLEVEL%
)

echo.
echo === Building tests ===
cmake --build "%BUILD_DIR%" --target test_rice_entropy test_timecode_roundtrip test_bgra_convert test_prores_encode test_perf_benchmark -j4

if %ERRORLEVEL% neq 0 (
  echo === Build FAILED ===
  exit /b %ERRORLEVEL%
)

echo.
echo === Running test_rice_entropy ===
"%BUILD_DIR%\test_rice_entropy.exe" --verbose

echo.
echo === Running test_timecode_roundtrip ===
"%BUILD_DIR%\test_timecode_roundtrip.exe" --verbose

echo.
echo === Running test_bgra_convert ===
"%BUILD_DIR%\test_bgra_convert.exe"

echo.
echo === Running test_prores_encode ===
"%BUILD_DIR%\test_prores_encode.exe"

echo.
echo === All done ===
