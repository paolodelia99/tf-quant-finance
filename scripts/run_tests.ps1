# Set the base directory and source the virtual environment creation script
. "$PSScriptRoot/create_venv.ps1"

# Collect all arguments except script name
$extraArgs = $args

Write-Host "Running tff tests"

if ($extraArgs -contains '--cov') {
    coverage run --source=tf_quant_finance -m pytest --no-header -vv
    coverage report -m
} else {
    pytest --no-header -vv
}
