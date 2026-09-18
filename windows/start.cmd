@echo off
setlocal
rem Enter your TSC wallet below, then double-click this file.
set "WALLET=YOUR_TSC_WALLET"
set "WORKER=windows"
set "DEVICES=0"
if "%WALLET%"=="YOUR_TSC_WALLET" (
  echo Edit start.cmd and set WALLET to your own TSC address first.
  pause
  exit /b 2
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" -Wallet "%WALLET%" -Worker "%WORKER%" -Devices "%DEVICES%"
pause
