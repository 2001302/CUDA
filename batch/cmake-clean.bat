@echo off
REM CMake: Clean
cd /d "%~dp0.."
cmake --build build --target clean
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

