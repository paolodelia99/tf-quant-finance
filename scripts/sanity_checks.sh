#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/create_venv.sh"

pip install black

#echo "Running the linter"

# pylint tf_quant_finance -output-format=text:pylint_res.txt,colorized

echo "Running the code formatter"

black --check tf_quant_finance