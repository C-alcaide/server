@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

cl 2>&1 | findstr /i "version"
echo.

set BUILD_DIR=d:\Github\CasparCG-cuda\tests_standalone\build
set SRC_DIR=d:\Github\CasparCG-cuda\tests_standalone

echo === Configuring ===
cmake -S "%SRC_DIR%" -B "%BUILD_DIR%" ^
  -G "Ninja" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler -Wno-deprecated-gpu-targets" ^
  -DCMAKE_C_COMPILER=cl ^
  -DCMAKE_CXX_COMPILER=cl

if %ERRORLEVEL% neq 0 (
    echo === Configure FAILED ===
    exit /b %ERRORLEVEL%
)

echo.
echo === Building decklink_prores_capture ===
cmake --build "%BUILD_DIR%" --target decklink_prores_capture -j4

if %ERRORLEVEL% neq 0 (
    echo === Build FAILED ===
    exit /b %ERRORLEVEL%
)

echo.
echo === Build successful ===
echo   Exe: %BUILD_DIR%\decklink_prores_capture.exe
echo.
echo   This .exe is self-contained: static CRT + static CUDA runtime.
echo   Copy it to any Windows PC with DeckLink Desktop Video installed.
echo.
echo Usage examples:
echo   decklink_prores_capture.exe                             (all cards, HQ, 240 s)
echo   decklink_prores_capture.exe --count 4 --profile hq --duration 240
echo   decklink_prores_capture.exe --devices "DeckLink 1,DeckLink 2" --profile std
echo   decklink_prores_capture.exe --count 1 --profile proxy --duration 60
echo.
echo CPU comparison (run on a spare card simultaneously):
echo   ffmpeg -f decklink -i "DeckLink X" -t 240 -c:v prores_ks -profile:v hq -y cpu_hq.mov

