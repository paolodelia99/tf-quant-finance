#!/bin/bash

COVERAGE="FALSE"

POSITIONAL_ARGS=()

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -cov|--coverage)
            COVERAGE="TRUE"
            shift 2
            ;;
        -*|--*)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
    esac
done

# Setting the basedir
source "$(dirname "${BASH_SOURCE[0]}")/create_venv.sh"

pip install pytest

# Test suites to run
echo "Compiling proto files!"
protoc --python_out=. --proto_path=. tf_quant_finance/experimental/pricing_platform/instrument_protos/*.proto

echo "Running tff tests"

# Collect all arguments except script name
extra_args=("${@:1}")

if [[ $COVERAGE = "TRUE" ]]; then
    pip install coverage
    coverage run --source=tf_quant_finance -m pytest --no-header -vv
    coverage xml
else
  pytest --no-header -vv
fi
