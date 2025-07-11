#!/bin/bash

# Setting the basedir
source "$(dirname "${BASH_SOURCE[0]}")/create_venv.sh"

pip install pytest

# Test suites to run
tests=("tests/black_scholes" "tests/datetime" "tests/utils" "tests/experimental")

echo "Compiling proto files!"
protoc --python_out=. --proto_path=. tf_quant_finance/experimental/pricing_platform/instrument_protos/*.proto

echo "Running tff tests"
pytest --no-header -vv "${tests[@]}"
