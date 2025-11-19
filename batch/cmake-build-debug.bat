@echo off
REM CMake: Build (Debug)
cd /d "%~dp0.."
cmake --build build --config Debug
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

