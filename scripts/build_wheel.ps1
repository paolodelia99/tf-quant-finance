# Build script for tf-quant-finance using Python build tools (PowerShell version)
# This replaces the Bazel-based build process

. "$PSScriptRoot/create_venv.ps1"

$DEST_DIR = "dist"
$NIGHTLY_BUILD = $false

# Parse command line arguments
foreach ($arg in $args) {
    if ($arg -eq "--nightly") {
        Write-Host "Building a nightly build."
        $NIGHTLY_BUILD = $true
    } else {
        $DEST_DIR = $arg
    }
}

# Create destination directory
if (-not (Test-Path $DEST_DIR)) {
    New-Item -ItemType Directory -Path $DEST_DIR | Out-Null
}
$DEST_DIR = Resolve-Path $DEST_DIR
Write-Host "=== Destination directory: $DEST_DIR"

# Clean previous builds
Write-Host "=== Cleaning previous builds"
Remove-Item -Recurse -Force build, dist, *.egg-info -ErrorAction SilentlyContinue

Write-Host "=== Building wheel with python -m build"

# Check if build module is available
$buildCheck = & python -c "import build" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: 'build' module not found. Please install it with:"
    Write-Host "pip install build"
    exit 1
}

if ($NIGHTLY_BUILD) {
    # For nightly builds, pass the --nightly flag to setup.py
    python setup.py bdist_wheel --nightly
    Move-Item dist\*.whl "$DEST_DIR\"
} else {
    python -m build --wheel --outdir "$DEST_DIR"
}

Write-Host "=== Build completed successfully"
Write-Host "=== Output wheel file is in: $DEST_DIR"
Get-ChildItem "$DEST_DIR\*.whl" | Format-List
