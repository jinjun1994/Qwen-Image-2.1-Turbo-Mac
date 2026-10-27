#!/bin/sh
set -eu
mqwen_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$mqwen_root"
uv venv --python 3.13 --allow-existing .venv
uv pip install --python .venv/bin/python -e '.[native]'
