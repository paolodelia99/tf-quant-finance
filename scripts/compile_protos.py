#!/usr/bin/env python3
"""
Compile protocol buffer files for tf-quant-finance.
This script replaces the protobuf compilation that was handled by Bazel.
"""
from contextlib import chdir
import subprocess
import sys
from pathlib import Path


def compile_protos():
    """Compile all .proto files to Python _pb2.py files."""
    proto_dir = Path("tf_quant_finance/experimental/pricing_platform/instrument_protos")
    
    if not proto_dir.exists():
        print(f"Protocol buffer directory {proto_dir} does not exist.")
        return True
    
    with chdir(proto_dir):
        project_root = Path(".")
        proto_files = list(project_root.glob("*.proto"))

        if not proto_files:
            print("No .proto files found.")
            return True
    
        # Check if protoc is available
        try:
            subprocess.run(["protoc", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: protoc (Protocol Buffer Compiler) not found.")
            print("Install with:")
            print("  Ubuntu/Debian: sudo apt-get install protobuf-compiler")
            print("  macOS: brew install protobuf")
            print("  Or download from: https://github.com/protocolbuffers/protobuf/releases")
            return False
    
        print(f"Compiling {len(proto_files)} protocol buffer files...")
    
        success = True
        
        # Compile all proto files in one go to handle dependencies
        all_proto_files = [str(f) for f in proto_files]
        
        cmd = [
            "protoc",
            f"--python_out={project_root}",
            f"--proto_path={project_root}",
        ] + all_proto_files
    
        try:
            _ = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✓ Compiled all protocol buffer files successfully")
            
            # List the generated files
            pb2_files = list(project_root.glob("*_pb2.py"))
            print(f"Generated {len(pb2_files)} Python files:")
            for pb2_file in pb2_files:
                print(f"  - {pb2_file}")
                
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to compile protocol buffer files")
            print(f"  Error: {e.stderr}")
            success = False
    
    return success


if __name__ == "__main__":
    success = compile_protos()
    sys.exit(0 if success else 1)
