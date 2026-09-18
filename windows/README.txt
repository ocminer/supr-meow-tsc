supr-meow-tsc 0.7.0 - native Windows x64 / NVIDIA CUDA

This package runs natively on Windows; WSL and Docker are not required.
Use Windows 10/11 x64 and an NVIDIA Ampere or newer GPU with a driver
compatible with CUDA 13.3. Install the Microsoft Visual C++ 2015-2022
x64 runtime if Windows reports a missing VCRUNTIME/MSVCP DLL.

1. Extract the complete ZIP to a writable directory.
2. Edit start.cmd and set WALLET to your own TSC wallet.
3. Double-click start.cmd. The pinned Q8 model downloads on first start.
   Keep the models directory (about 9 GB) for later starts.

PowerShell example:
  .\start.ps1 -Wallet YOUR_TSC_WALLET -Worker rig1 -Devices 0

Two matching cards sharing one model:
  .\start.ps1 -Wallet YOUR_TSC_WALLET -Devices '0,1' -SplitModel

Stop your old miner first. Use the standard Q8 profile and F16 KV cache;
it identifies itself to the pool as supr-meow-tsc/0.7.0 and emits proof v4.
The launcher uses tsc.suprnova.cc:3310 unless you pass -Pool.
CPU proof and native socket tests run in GitHub CI. The hosted Windows
runner has no NVIDIA GPU; Windows GPU mining is not hardware-tested there.
Linux release packages have been tested mining with live v4 verification.

BUILD-INFO.json records the source and CI run. SHA256SUMS.txt covers files
inside this package; the external .sha256 file covers the complete ZIP.
Source and build instructions: https://github.com/ocminer/supr-meow-tsc
