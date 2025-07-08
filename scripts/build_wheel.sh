#!/usr/bin/env bash
# Build script for tf-quant-finance using Python build tools

source "$(dirname "${BASH_SOURCE[0]}")/create_venv.sh"

set -e

function main() {
    DEST_DIR="dist"
    NIGHTLY_BUILD=false
    
    # Parse command line arguments
    while [[ ! -z "${1}" ]]; do
        case ${1} in
            "--nightly")
                echo "Building a nightly build."
                NIGHTLY_BUILD=true
                ;;
            *)
                DEST_DIR=${1}
                ;;
        esac
        shift
    done
    
    # Create destination directory
    mkdir -p ${DEST_DIR}
    DEST_DIR=$(readlink -f "${DEST_DIR}")
    echo "=== Destination directory: ${DEST_DIR}"
    
    # Clean previous builds
    echo "=== Cleaning previous builds"
    rm -rf build/ dist/ *.egg-info/
    
    echo "=== Building wheel with python -m build"
    
    # Check if build module is available
    if ! python -c "import build" 2>/dev/null; then
        echo "Error: 'build' module not found. Please install it with:"
        echo "pip install build"
        exit 1
    fi
    
    if [ "$NIGHTLY_BUILD" = true ]; then
        # For nightly builds, pass the --nightly flag to setup.py
        python setup.py bdist_wheel --nightly
        mv dist/*.whl "${DEST_DIR}/"
    else
        python -m build --wheel --outdir "${DEST_DIR}"
    fi
    
    echo "=== Build completed successfully"
    echo "=== Output wheel file is in: ${DEST_DIR}"
    ls -la "${DEST_DIR}"/*.whl
}

main "$@"
