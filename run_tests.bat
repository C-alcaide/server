@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

set BUILD_DIR=d:\Github\CasparCG-cuda\tests_standalone\build

cmake --build "%BUILD_DIR%" --target test_rice_entropy test_bgra_convert test_perf_benchmark test_prores_encode test_timecode_roundtrip -j4

if %ERRORLEVEL% neq 0 (
  echo === Build FAILED ===
  exit /b %ERRORLEVEL%
)

echo.
echo === Running test_rice_entropy ===
"%BUILD_DIR%\test_rice_entropy.exe" --verbose
echo.
echo === Running test_bgra_convert ===
"%BUILD_DIR%\test_bgra_convert.exe"
echo.
echo === Running test_timecode_roundtrip ===
"%BUILD_DIR%\test_timecode_roundtrip.exe" "%TEMP%"
echo.
echo === All tests done ===
