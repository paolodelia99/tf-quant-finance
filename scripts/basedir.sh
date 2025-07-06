#!/bin/bash

# Get the directory of the current script, resolving symlinks
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Go one level up to get the base project directory
BASEDIR="$(dirname "$SCRIPT_DIR")"

# Print the base directory
echo "Basedir set to: $BASEDIR"
