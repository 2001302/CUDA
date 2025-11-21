@echo off
REM Run example2 (Release)
cd /d "%~dp0.."
build\bin\Release\example2.exe
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%


