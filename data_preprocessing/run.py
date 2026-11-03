"""Preprocess already downloaded data for one supported dataset."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['taobao', 'merrec', 'taobao_ads', 'tecd'])
    parser.add_argument('--raw-root', type=Path, help='Existing downloaded source directory')
    parser.add_argument('--out-root', type=Path, help='Converted or prepared dataset directory')
    parser.add_argument('--dry-run', action='store_true', help='Print the preprocessing command without writing data')
    args, extra = parser.parse_known_args()
    if any(x.split('=')[0] in {'--download', '--skip-download', '--raw-root', '--out-root'} for x in extra):
        parser.error('Download separately with scripts/download.sh; preprocessing reads local files')
    base = Path(os.environ.get('BENCHMARK_ROOT', 'benchmark')).expanduser().resolve() / args.dataset
    raw = (args.raw_root or Path(os.environ.get('RAW_ROOT', str(base / 'raw')))).expanduser().resolve()
    default_out = base / ('converted' if args.dataset in {'taobao', 'merrec'} else 'prepared')
    output = (args.out_root or Path(os.environ.get('DATA_ROOT', str(default_out)))).expanduser().resolve()
    if not args.dry_run and not raw.is_dir():
        parser.error(f'Source directory missing: {raw}. Run scripts/download.sh {args.dataset} first')
    command = [sys.executable, '-m', f'data_preprocessing.{args.dataset}.convert',
               '--raw-root', str(raw), '--out-root', str(output), *extra]
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
