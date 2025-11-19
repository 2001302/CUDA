@echo off
REM CMake: Rebuild (Debug)
cd /d "%~dp0.."
cmake --build build --config Debug --clean-first
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

