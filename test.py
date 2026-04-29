#!/usr/bin/env python3
"""
test.py
-------
Reproduces every figure in submission/figures/ by running all .py scripts
found in submission/code/.

Usage (from repository root):
    python test.py

For each .py file in submission/code/:
  1. Executes the script with the current Python interpreter.
  2. On success prints "OK"; on failure prints the captured stderr/stdout.

After all scripts have run, lists every PDF present in submission/figures/
together with its file size so the caller can confirm all figures were written.

Exit code:
  0  — all scripts completed without error
  1  — one or more scripts failed
"""
import os
import subprocess
import sys

REPO_ROOT   = os.path.dirname(os.path.abspath(__file__))
CODE_DIR    = os.path.join(REPO_ROOT, 'submission', 'code')
FIGURES_DIR = os.path.join(REPO_ROOT, 'submission', 'figures')


def main() -> int:
    # Verify the code directory exists
    if not os.path.isdir(CODE_DIR):
        print(f'ERROR: code directory not found: {CODE_DIR}', file=sys.stderr)
        return 1

    # Ensure the figures output directory exists
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Collect all .py files in submission/code/, sorted for deterministic order
    py_files = sorted(f for f in os.listdir(CODE_DIR) if f.endswith('.py'))

    if not py_files:
        print('No .py files found in submission/code/', file=sys.stderr)
        return 1

    print(f'Found {len(py_files)} script(s) in {CODE_DIR}')
    print('-' * 60)

    failures = []

    for fname in py_files:
        fpath = os.path.join(CODE_DIR, fname)
        print(f'  {fname} ... ', end='', flush=True)

        result = subprocess.run(
            [sys.executable, fpath],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            # Print the script's own output on the same line
            script_out = result.stdout.strip()
            print(f'OK  {script_out}')
        else:
            print('FAILED')
            if result.stdout.strip():
                for line in result.stdout.strip().splitlines():
                    print(f'    stdout: {line}')
            if result.stderr.strip():
                for line in result.stderr.strip().splitlines():
                    print(f'    stderr: {line}')
            failures.append(fname)

    # Report generated PDFs
    print()
    print('-' * 60)
    if os.path.isdir(FIGURES_DIR):
        pdfs = sorted(f for f in os.listdir(FIGURES_DIR) if f.endswith('.pdf'))
        print(f'Figures in {FIGURES_DIR} ({len(pdfs)} file(s)):')
        for p in pdfs:
            size_kb = os.path.getsize(os.path.join(FIGURES_DIR, p)) / 1024
            print(f'  {p:<40s}  {size_kb:6.1f} KB')
    else:
        print(f'WARNING: figures directory not found: {FIGURES_DIR}')

    print()
    if failures:
        print(f'RESULT: {len(failures)} script(s) FAILED — {failures}')
        return 1
    else:
        print(f'RESULT: all {len(py_files)} scripts completed successfully.')
        return 0


if __name__ == '__main__':
    sys.exit(main())
