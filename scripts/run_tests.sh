#!/bin/bash

# Setting the basedir
source "$(dirname "${BASH_SOURCE[0]}")/create_venv.sh"

pip install pytest

# Test suites to run
tests=("tests/black_scholes" "tests/datetime" "tests/utils")

echo "Running tff tests"
pytest --no-header -vv "${tests[@]}"
