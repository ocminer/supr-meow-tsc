param(
    [Parameter(Mandatory=$true)][string]$Wallet,
    [string]$Worker = 'windows',
    [string]$Pool = 'stratum+tcp://tsc.suprnova.cc:3310',
    [string]$Devices = '0',
    [string]$ModelDir = "$PSScriptRoot/models",
    [switch]$SplitModel
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$model = Join-Path $ModelDir 'Qwen3-8B-Q8_0-full-v2.gguf'
$expected = '30f3a9df384a08453ff0bf5715489e174e986cd033af2a9430e4ddb039cd73c7'
if (!(Test-Path $model)) {
    New-Item -ItemType Directory -Force $ModelDir | Out-Null
    $part = "$model.download-$PID"
    try {
        & curl.exe --fail --location --retry 5 --proto '=https' --proto-redir '=https' `
            --output $part 'https://huggingface.co/luckypoolio/Qwen3-8B-Q8_0-full-v2.gguf/resolve/850bbb3a59d5c855c5d7b78831ff823ca1791cfb/Qwen3-8B-Q8_0-full-v2.gguf'
        if ($LASTEXITCODE) { throw 'Model download failed' }
        if ((Get-FileHash $part -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) { throw 'Model checksum mismatch' }
        Move-Item $part $model
    } finally { if (Test-Path $part) { Remove-Item $part } }
}
if ((Get-FileHash $model -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) { throw 'Wrong model; select the pinned Q8 full-v2 model' }
$env:MEOW_DOUBLE_BUFFER = '0'
$env:POW_PROMPT_STYLE = '1'
$minerArgs = @('--q8', '--kv-cache', 'f16', '-o', $Pool, '-u', "$Wallet.$Worker", '-d', $Devices, '--model', $model)
if ($SplitModel) { $minerArgs += '--split-model' }
& "$PSScriptRoot/supr-meow-tsc.exe" @minerArgs
exit $LASTEXITCODE
