#!/bin/bash

COVERAGE="FALSE"

POSITIONAL_ARGS=()

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -cov|--coverage)
            COVERAGE="TRUE"
            shift
            ;;
        -*|--*)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Setting the basedir
source "$(dirname "${BASH_SOURCE[0]}")/create_venv.sh"

pip install pytest

# Compile protos files
echo "Compiling proto files!"
protoc --python_out=. --proto_path=. tf_quant_finance/experimental/pricing_platform/instrument_protos/*.proto
echo "Generated proto files"
ls -ld tf_quant_finance/experimental/pricing_platform/instrument_protos/*.proto


# Run tests
if [[ $COVERAGE = "TRUE" ]]; then
    pip install coverage
    echo "Running tff tests with Coverage"
    coverage run --source=tf_quant_finance -m pytest --no-header -vv
    coverage xml
else
    echo "Running tff tests"
    pytest --no-header -vv
fi
