param([ValidateRange(1024,65535)][int]$Port=8777)
$ErrorActionPreference='Stop'
$runtimePython=Join-Path $env:USERPROFILE '.cache/pulsemorph-agentic/envs/app-public-win-cp311-v1/Scripts/python.exe'
if (-not(Test-Path -LiteralPath $runtimePython -PathType Leaf)) {throw 'Prepared public Python unavailable; no installation attempted.'}
$projectRoot=(Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$bootstrap="import sys;from pathlib import Path;root=Path(sys.argv.pop(1));sys.path.insert(0,str(root/'src'));from pulsemorph.agentic.app.phrase.entrypoint_v2 import prepare_parent_imports;prepare_parent_imports(root);from pulsemorph.agentic.embodied.app.server_v1 import main;main()"
& $runtimePython -I -B -c $bootstrap $projectRoot --trusted-local --port $Port
exit $LASTEXITCODE
