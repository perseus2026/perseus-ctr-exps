"""Download source files for one supported dataset without preparing training data."""
import argparse
import os
from pathlib import Path


def download(dataset, raw):
    if dataset in {'taobao', 'merrec'}:
        from huggingface_hub import snapshot_download
        repository = 'salmon1802/' + {'taobao': 'Taobao', 'merrec': 'MerRec'}[dataset]
        snapshot_download(repo_id=repository, repo_type='dataset', local_dir=raw)
    elif dataset == 'taobao_ads':
        from .taobao_ads import _download, _history
        _download.collect(argparse.Namespace(output=raw / 'impressions'))
        _history.collect(argparse.Namespace(output=raw / 'history'))
    else:
        from .tecd import _download
        _download.collect(argparse.Namespace(
            output=raw, domains=['retail', 'marketplace', 'offers', 'payments'],
            small_days=None, payments_days=None,
        ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['taobao', 'merrec', 'taobao_ads', 'tecd'])
    parser.add_argument('--raw-root', type=Path, help='Destination for source files')
    parser.add_argument('--dry-run', action='store_true', help='Print the source and destination without downloading')
    args = parser.parse_args()
    base = Path(os.environ.get('BENCHMARK_ROOT', 'benchmark')).expanduser().resolve() / args.dataset
    raw = (args.raw_root or Path(os.environ.get('RAW_ROOT', str(base / 'raw')))).expanduser().resolve()
    sources = {'taobao': 'https://huggingface.co/datasets/salmon1802/Taobao',
               'merrec': 'https://huggingface.co/datasets/salmon1802/MerRec',
               'taobao_ads': 'HF Taobao Ads tables and Kaggle behaviour archive',
               'tecd': 'https://huggingface.co/datasets/t-tech/T-ECD'}
    print(f'Download {sources[args.dataset]} → {raw}', flush=True)
    if not args.dry_run:
        download(args.dataset, raw)


if __name__ == '__main__':
    main()
