param(
    [ValidateRange(1024,65535)][int]$Port = 8774,
    [string]$RuntimePython = (Join-Path $env:USERPROFILE '.cache/pulsemorph-agentic/envs/app-public-win-cp311-v1/Scripts/python.exe')
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $RuntimePython -PathType Leaf)) { throw 'Reviewed runtime Python is unavailable; no installation was attempted.' }
$entry = Join-Path $PSScriptRoot 'run_expression_studio_v3.py'
if (-not (Test-Path -LiteralPath $entry -PathType Leaf)) { throw 'Expression Studio V3 launcher is unavailable.' }
& $RuntimePython -I -B $entry --trusted-local --port $Port
exit $LASTEXITCODE
