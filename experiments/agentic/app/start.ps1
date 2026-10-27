param([int]$Port = 8765, [string]$RuntimePython)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$env:PYTHONPATH = (Join-Path $projectRoot 'src') + ';' + $projectRoot + ';' + (Join-Path $projectRoot 'experiments/agentic')
if ($RuntimePython) {
    if (-not (Test-Path -LiteralPath $RuntimePython -PathType Leaf)) {
        throw 'RuntimePython must name an existing local Python executable.'
    }
    $selectedPython = (Resolve-Path -LiteralPath $RuntimePython).Path
    $launchCode = 'import runpy,sys; root=sys.argv.pop(1); sys.path[:0]=[root+"/src",root,root+"/experiments/agentic"]; sys.argv[0]="pulsemorph.agentic.app.server"; runpy.run_module("pulsemorph.agentic.app.server",run_name="__main__")'
    & $selectedPython -I -c $launchCode $projectRoot --trusted-local --port $Port --runtime-python $selectedPython
} else {
    & (Join-Path $projectRoot '.venv-motor/Scripts/python.exe') -m pulsemorph.agentic.app.server --trusted-local --port $Port
}
