@echo off
REM CMake: Rebuild (Release)
cd /d "%~dp0.."
cmake --build build --config Release --clean-first
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

