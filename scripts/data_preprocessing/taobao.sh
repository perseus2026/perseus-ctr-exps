#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$root"
exec "${PYTHON:-python3}" -m data_preprocessing.run taobao "$@"
