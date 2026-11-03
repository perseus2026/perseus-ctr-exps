"""Prepare the selected recipe if needed, then train and evaluate it."""
import argparse
import os
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(command, dry_run):
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def main(*, lcn=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['taobao_ads', 'tecd'] if lcn else ['taobao', 'merrec', 'taobao_ads', 'tecd'])
    if not lcn:
        parser.add_argument('model', nargs='?', default='est')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--history', choices=['single', 'cross'], default='cross')
    parser.add_argument('--config', type=Path, help='Defaults to CONFIG_PATH or the selected baseline recipe')
    parser.add_argument('--data-root', type=Path, help='Converted Grid 1/2 data or prepared cross-domain data')
    parser.add_argument('--prepared', type=Path, help='Existing prepared Grid 1/2 data; otherwise prepared automatically')
    parser.add_argument('--workdir', type=Path, help='New run directory; defaults to WORKDIR or benchmark/<dataset>/runs/...')
    parser.add_argument('--stores', type=Path, help='Taobao LCN domain-separated stores')
    parser.add_argument('--processes', type=int)
    parser.add_argument('--microbatch', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--precision', choices=['no', 'fp16', 'bf16'])
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Print preparation and training commands without writing outputs')
    args = parser.parse_args()
    if args.seed is None:
        try:
            args.seed = int(os.environ.get('SEED', 42))
        except ValueError:
            parser.error('SEED must be an integer, or supply --seed explicitly')
    for flag, value, minimum in [('processes', args.processes, 1),
                                  ('microbatch', args.microbatch, 1),
                                  ('workers', args.workers, 0)]:
        if value is not None and value < minimum:
            parser.error(f'--{flag} must be at least {minimum}')
    if args.cpu and args.processes not in (None, 1):
        parser.error('CPU training supports one process')
    model = 'lcn' if lcn else args.model
    cross = args.dataset in {'taobao_ads', 'tecd'}
    allowed = ({'lcn'} if lcn else {'perseus', 'est', 'hemix', 'hyformer'}) if cross else {'est', 'hyformer', 'mixformer', 'tokenmixer', 'hiformer', 'hemix'}
    if model not in allowed:
        parser.error(f'Unsupported model for {args.dataset}: {model}')
    if cross:
        default_config = ROOT / 'configs/cross_domain' / args.dataset / f'{model}_{args.history}.yaml'
    else:
        default_config = ROOT / 'configs/baseline_comparison' / args.dataset / f'ctr-v1-{args.dataset}-reproduce-{model}.yaml'
    config = args.config or Path(os.environ.get('CONFIG_PATH', str(default_config)))
    config = config.expanduser().resolve()
    if not config.is_file():
        parser.error(f'Configuration missing: {config}')
    base = Path(os.environ.get('BENCHMARK_ROOT', 'benchmark')).expanduser().resolve() / args.dataset
    default_data = base / ('prepared' if cross else 'converted')
    data = (args.data_root or Path(os.environ.get('DATA_ROOT', str(default_data)))).expanduser().resolve()
    recipe_name = config.stem
    component_root = ROOT / 'configs/component_substitutions' / args.dataset
    if config.is_relative_to(component_root):
        recipe_name = '-'.join(config.relative_to(component_root).with_suffix('').parts)
    suffix = f'{model}-{args.history}-s{args.seed}' if cross else f'{recipe_name}-s{args.seed}'
    workdir = (args.workdir or Path(os.environ.get('WORKDIR', str(base / 'runs' / suffix)))).expanduser().resolve()
    if workdir.exists():
        parser.error(f'Run directory already exists: {workdir}')
    if cross:
        if args.prepared is not None or any(v is not None for v in [args.processes, args.microbatch, args.workers, args.precision]) or args.cpu:
            parser.error('Cross-domain runs use the hardware/training settings in their YAML; Grid 1/2 overrides do not apply')
        expected_seeds = range(42, 52) if args.dataset == 'taobao_ads' else range(42, 47)
        if args.seed not in expected_seeds:
            parser.error(f'This study uses seeds {expected_seeds.start}–{expected_seeds.stop - 1}')
        if args.stores is not None and not (lcn and args.dataset == 'taobao_ads'):
            parser.error('--stores is only used for Taobao Ads LCN')
        command = [sys.executable, '-m', 'perseus_lib.cross_domain.run',
                   '--dataset', args.dataset, '--model', model, '--history', args.history,
                   '--seed', str(args.seed), '--data-root', str(data),
                   '--config', str(config), '--output', str(workdir)]
        if lcn and args.dataset == 'taobao_ads':
            stores = (args.stores or base / 'lcn-stores').expanduser().resolve()
            if not (stores / 'target/data.arrow').is_file() or not (stores / 'source/data.arrow').is_file():
                run([sys.executable, '-m', 'baselines.lcn.prepare_lcn_stores',
                     '--workdir', str(data / 'cross'), '--output', str(stores), '--target-prefix', 'target-'], args.dry_run)
            command += ['--stores', str(stores)]
        run(command, args.dry_run)
    else:
        if args.stores is not None:
            parser.error('--stores is only used for Taobao Ads LCN')
        if args.cpu:
            os.environ['ACCELERATE_USE_CPU'] = 'true'
        import yaml
        try:
            training = yaml.safe_load(config.read_text())['training']
            target_batch = int(training['dataloader']['batch_size']) * int(training.get('accumulation') or 1)
            microbatch = args.microbatch if args.microbatch is not None else int(training['dataloader']['batch_size'])
        except (ValueError, TypeError, KeyError) as error:
            parser.error(f'Invalid batch settings in configuration: {error}')
        processes = args.processes if args.processes is not None else 1
        if target_batch < 1 or microbatch < 1 or target_batch % (microbatch * processes):
            parser.error('Effective batch must be a positive multiple of microbatch × processes')
        prepared = (args.prepared or base / 'prepared' / recipe_name).expanduser().resolve()
        if not (prepared / 'checkpoint/config.yaml').is_file() or not (prepared / 'dataset').is_dir():
            if args.prepared is not None:
                parser.error(f'Explicit prepared directory is incomplete: {prepared}')
            run([sys.executable, '-m', 'perseus_lib.prepare', '--config', str(config),
                 '--converted', str(data), '--workdir', str(prepared)], args.dry_run)
        metadata = prepared / 'preparation.json'
        if args.prepared is None and metadata.is_file():
            recorded = json.loads(metadata.read_text())
            if Path(recorded['converted']).resolve() != data.resolve():
                parser.error('Prepared data came from another converted root; choose a new preparation directory with --prepared after preparing it explicitly')
        command = [sys.executable, '-m', 'perseus_lib.train', '--config', str(config),
                   '--prepared', str(prepared), '--workdir', str(workdir), '--seed', str(args.seed)]
        for flag, value in [('processes', args.processes), ('microbatch', args.microbatch),
                            ('workers', args.workers), ('precision', args.precision)]:
            if value is not None:
                command += ['--' + flag, str(value)]
        if args.cpu:
            command.append('--cpu')
        run(command, args.dry_run)


if __name__ == '__main__':
    main()
