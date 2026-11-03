#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
library=${1:-"$root/../repo"}
if [ ! -f "$library/pyproject.toml" ]; then
    echo "Usage: $0 /path/to/perseus-ctr (defaults to sibling ../repo)" >&2
    exit 2
fi
"${PYTHON:-python3}" -m pip install -e "$library"
"${PYTHON:-python3}" -m pip install -r "$root/requirements.txt"
