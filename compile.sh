#!/usr/bin/env bash
# Python is interpreted; "compilation" only ensures the script is executable
# and that a usable Python 3 is present.
set -e

chmod +x cpg.py
python3 -c "import sys; assert sys.version_info >= (3, 7), 'Python >= 3.7 required'"
