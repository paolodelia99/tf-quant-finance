#!/bin/bash

# Setting the basedir
source "$(dirname "${BASH_SOURCE[0]}")/basedir.sh"

PYTHON=$(which python3)
VENV_DIR="$BASEDIR/venv"

function activate_venv() {
    . $VENV_DIR/bin/activate
}

if [ -d "$VENV_DIR" ]; then
    
    if [ -z $VIRTUAL_ENV ]; then
        echo "Virtual environment activated"
        activate_venv
    else
        echo "Virtual environment already active"
    fi


else
    # Create venv
    echo "Creating Virtual environment"
    $PYTHON -m venv $VENV_DIR

    . $VENV_DIR/bin/activate

    pip install -r $BASEDIR/requirements/build.txt

fi
