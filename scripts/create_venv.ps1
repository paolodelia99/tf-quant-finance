# Set the base directory
. "$PSScriptRoot/basedir.ps1"

$python = Get-Command python3 -ErrorAction SilentlyContinue
if (-not $python) {
    $python = Get-Command python
}
$venvDir = Join-Path $BaseDir "venv"

function Activate-Venv {
    & "$venvDir/Scripts/Activate.ps1"
}

if (Test-Path $venvDir) {
    if (-not $env:VIRTUAL_ENV) {
        Write-Host "Virtual environment activated"
        Activate-Venv
    } else {
        Write-Host "Virtual environment already active"
    }
} else {
    # Create venv
    Write-Host "Creating Virtual environment"
    & $python -m venv $venvDir

    & "$venvDir/Scripts/Activate.ps1"

    pip install -r (Join-Path $BaseDir "requirements/build.txt")
}
