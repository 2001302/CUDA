@echo off
REM CMake: Configure
cd /d "%~dp0.."
cmake -B build -S . -G "Visual Studio 17 2022" -A x64
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

