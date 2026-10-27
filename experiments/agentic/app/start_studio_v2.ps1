param([int]$Port = 8765, [Parameter(Mandatory=$true)][string]$RuntimePython)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
if (-not (Test-Path -LiteralPath $RuntimePython -PathType Leaf)) {
    throw 'RuntimePython must name an existing local Python executable.'
}
$studioPython = (Resolve-Path -LiteralPath $RuntimePython).Path
$launchCode = 'import runpy,sys; root=sys.argv.pop(1); sys.path[:0]=[root+"/src",root,root+"/experiments/agentic"]; sys.argv[0]="pulsemorph.agentic.app.phrase.entrypoint_v2"; runpy.run_module("pulsemorph.agentic.app.phrase.entrypoint_v2",run_name="__main__")'
& $studioPython -I -B -c $launchCode $projectRoot --trusted-local --port $Port --runtime-python $studioPython
