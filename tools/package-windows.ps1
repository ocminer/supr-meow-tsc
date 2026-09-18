$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$stage = Join-Path $env:RUNNER_TEMP 'supr-meow-tsc-0.7.0-windows-x86_64'
New-Item -ItemType Directory -Force $stage | Out-Null
$exe = Get-ChildItem build/windows -Filter supr-meow-tsc.exe -Recurse | Select-Object -First 1
if (!$exe) { throw 'Native miner executable missing' }
Copy-Item $exe.FullName "$stage/supr-meow-tsc.exe"
Get-ChildItem build/windows/bin -Filter '*.dll' -Recurse | Copy-Item -Destination $stage
Get-ChildItem build/windows/vcpkg_installed/x64-windows/bin -Filter '*.dll' | Copy-Item -Destination $stage
Get-ChildItem "$env:CUDA_PATH/bin" -Filter '*.dll' -Recurse |
    Where-Object { $_.Name -match '^(cudart|cublas)' } | Copy-Item -Destination $stage
Copy-Item windows/* $stage
Copy-Item LICENSE,NOTICE $stage
$licenses = Join-Path $stage 'licenses'
New-Item -ItemType Directory -Force $licenses | Out-Null
foreach ($dep in @('gmp','openssl','argon2','zeromq','cppzmq','flatbuffers','boost-multiprecision')) {
    $copyright = "build/windows/vcpkg_installed/x64-windows/share/$dep/copyright"
    if (Test-Path $copyright) { Copy-Item $copyright "$licenses/$dep.txt" }
}
Copy-Item vendor/llama.cpp/LICENSE "$licenses/llama.txt"
Copy-Item vendor/tensorcash/shared-utils/chiavdf/LICENSE "$licenses/chiavdf.txt"
Copy-Item vendor/tensorcash/shared-utils/pow-utils/LICENSE "$licenses/tensorcash.txt"
Get-ChildItem $env:CUDA_PATH -Filter '*EULA*' -Recurse | Select-Object -First 1 | Copy-Item -Destination "$licenses/NVIDIA-CUDA-EULA.txt"
$info = @{
    version = '0.7.0'; source_commit = (& git rev-parse HEAD).Trim()
    ci_run = "https://github.com/$env:GITHUB_REPOSITORY/actions/runs/$env:GITHUB_RUN_ID"
    platform = 'native-windows-x86_64'; cuda = '13.3'; architectures = @(80,86,89,90,120)
    cpu_proof_and_socket_tests = 'passed'; windows_gpu_hardware_test = $false
}
$info | ConvertTo-Json | Set-Content "$stage/BUILD-INFO.json"
# Delay-load NVML permits --help without a GPU; all shipped runtime DLLs must resolve.
Push-Location $stage
try {
    & ./supr-meow-tsc.exe --help
    if ($LASTEXITCODE) { throw 'Packaged native executable cannot start' }
} finally { Pop-Location }
Get-ChildItem $stage -File -Recurse | Sort-Object FullName | ForEach-Object {
    $relative = $_.FullName.Substring($stage.Length + 1).Replace('\','/')
    "{0}  {1}" -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant(), $relative
} | Set-Content "$stage/SHA256SUMS.txt"
New-Item -ItemType Directory -Force dist | Out-Null
$zip = Join-Path $root 'dist/supr-meow-tsc-0.7.0-windows-x86_64.zip'
Compress-Archive -Path "$stage/*" -DestinationPath $zip -Force
"{0}  {1}" -f (Get-FileHash $zip -Algorithm SHA256).Hash.ToLowerInvariant(), (Split-Path $zip -Leaf) | Set-Content "$zip.sha256"
