@echo off
REM Run example1 (Release)
cd /d "%~dp0.."
build\bin\Release\example1.exe
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

