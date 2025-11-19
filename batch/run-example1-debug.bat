@echo off
REM Run example1 (Debug)
cd /d "%~dp0.."
build\bin\Debug\example1.exe
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

