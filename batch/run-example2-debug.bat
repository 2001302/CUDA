@echo off
REM Run example2 (Debug)
cd /d "%~dp0.."
build\bin\Debug\example2.exe
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%


