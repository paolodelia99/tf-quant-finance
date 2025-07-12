#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/create_venv.sh"

pip install pip-audit

#echo "Running the linter"

# pylint tf_quant_finance -output-format=text:pylint_res.txt,colorized

echo "Running the security audit for the build dependencies"

pip-audit -r requirements/build.txt