#!/usr/bin/env python3
"""
Simple helper script to execute a Jupyter notebook from the command line.
"""
import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path

def run_notebook(notebook_path: str):
    p = Path(notebook_path)
    if not p.exists():
        print(f"Error: Notebook {notebook_path} not found.")
        sys.exit(1)

    print(f"Executing {notebook_path}...")
    with open(p) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    try:
        ep.preprocess(nb, {'metadata': {'path': str(p.parent)}})
    except Exception as e:
        print(f"Error executing notebook: {e}")
        sys.exit(1)

    with open(p, 'w') as f:
        nbformat.write(nb, f)
    print(f"Done. Saved output to {notebook_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_notebook.py <notebook_path>")
        sys.exit(1)
    run_notebook(sys.argv[1])
