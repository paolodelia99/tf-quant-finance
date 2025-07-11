# Set the base directory and source the virtual environment creation script
. "$PSScriptRoot/create_venv.ps1"

# Test suites to run
$tests = @("tests/black_scholes", "tests/datetime", "tests/utils" "tests/experimental")

Write-Host "Running tff tests"
pytest -no-header -vv $tests
