"""Compare the tested numerical paths with an external UniRank checkout."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unirank-root', type=Path, required=True,
                        help='UniRank source checkout containing model_zoo/ and config/')
    parser.add_argument('--output', type=Path, default=Path('benchmark/audit'),
                        help='Directory for numerical reports (default: benchmark/audit)')
    args = parser.parse_args()
    reference = args.unirank_root.expanduser().resolve()
    for required in ['model_zoo/EST.py', 'config/model_config.yaml', 'config/dataset_config.yaml']:
        if not (reference / required).is_file():
            parser.error(f'UniRank source file missing: {reference / required}')
    os.environ['ACCELERATE_USE_CPU'] = 'true'
    import perseus
    runtime = Path(perseus.__file__).resolve().parent.parent
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    env = {**os.environ, 'UNIRANK_ROOT': str(reference), 'PERSEUS_ROOT': str(runtime),
           'PARITY_OUTPUT_DIR': str(output), 'PYTHONDONTWRITEBYTECODE': '1'}
    env['PYTHONPATH'] = str(runtime) + os.pathsep + env.get('PYTHONPATH', '')
    subprocess.run([sys.executable, str(scripts / 'full_parity.py'), '--out-dir', str(output)],
                   env=env, cwd=scripts, check=True)
    subprocess.run([sys.executable, str(scripts / 'core_parity.py'), '--reference', str(reference),
                    '--runtime', str(runtime), '--out', str(output / 'core.json')],
                   env=env, cwd=scripts, check=True)
    dense = json.loads((output / 'full_toy_equivalence.json').read_text())
    blocks = json.loads((output / 'core.json').read_text())
    keys = ['allclose', 'input_gradients_allclose', 'task_loss_allclose', 'projection_gradients_allclose']
    expected = {(ds, model, regime) for ds in ['taobao', 'merrec']
                for model in ['est', 'hyformer', 'mixformer', 'hemix']
                for regime in ['warm', 'mixed', 'cold']}
    actual = {(r['dataset'], r['family'], r['regime']) for r in dense['results']}
    passed = actual == expected and len(dense['results']) == len(expected)
    passed = passed and all(all(row[k] for k in keys) for row in dense['results'])
    passed = passed and blocks['status'] == 'PASS'
    summary = {'status': 'PASS' if passed else 'FAIL', 'dense_cases': len(dense['results']),
               'block_models': ['TokenMixer', 'HiFormer'],
               'scope': 'Aligned synthetic embeddings/weights, four dense paths and two internal blocks; CPU FP32. No data or full-training equivalence claim.',
               'library': str(runtime), 'unirank': str(reference),
               'reports': ['full_toy_equivalence.json', 'core.json']}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
