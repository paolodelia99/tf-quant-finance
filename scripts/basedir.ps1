# Get the directory of the current script, resolving symlinks
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

# Go one level up to get the base project directory
$BaseDir = Split-Path -Parent $ScriptDir

# Print the base directory
Write-Host "Basedir set to: $BaseDir"