#!/usr/bin/env bash
set -e
chmod +x cpg.py
python3 -c "import sys; assert sys.version_info >= (3, 7), 'Python >= 3.7 required'"
