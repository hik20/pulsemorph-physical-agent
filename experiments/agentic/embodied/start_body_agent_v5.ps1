param([Parameter(Mandatory=$true)][string]$Config,[Parameter(Mandatory=$true)][string]$RuntimePython)
$ErrorActionPreference='Stop'
$projectRoot=(Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$configPath=(Resolve-Path -LiteralPath $Config).Path
$pythonPath=(Resolve-Path -LiteralPath $RuntimePython).Path
$bootstrap="import sys;from pathlib import Path;root=Path(sys.argv.pop(1));sys.path.insert(0,str(root/'src'));from pulsemorph.agentic.embodied.app.server_v5 import main;main()"
& $pythonPath -I -B -c $bootstrap $projectRoot --trusted-local --config $configPath
exit $LASTEXITCODE
