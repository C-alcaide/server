@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cmake --build d:\Github\CasparVP\build --target ffmpeg decklink
echo EXIT_CODE=%ERRORLEVEL%
