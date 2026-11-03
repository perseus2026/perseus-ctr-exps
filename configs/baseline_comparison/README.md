# Baseline comparison (Grid 1)

Six architectures are available on Taobao and MerRec: EST, HyFormer, MixFormer, TokenMixer, HiFormer and HeMix.

```text
taobao/ctr-v1-taobao-reproduce-<model>.yaml
merrec/ctr-v1-merrec-reproduce-<model>.yaml
```

Replace `<model>` with its lowercase name. Each YAML stores the architecture, data-processing and training settings; seeds are command-line arguments.

Run from the repository root:

```sh
./scripts/download.sh taobao
./scripts/data_preprocessing/taobao.sh
./scripts/perseus.sh taobao est --seed 42
```

For MerRec:

```sh
./scripts/download.sh merrec
./scripts/data_preprocessing/merrec.sh
./scripts/perseus.sh merrec est --seed 42 --processes 2 --microbatch 4096 --workers 8
```

The second command prepares matching model data automatically before training. The MerRec example uses two processes and eight workers per process, preserving effective batch 32768. Repeat with a new seed for an independent run; do not copy the YAML.

Outputs are in `benchmark/<dataset>/runs/<configuration>-s<seed>/`, including `train.log`, checkpoints and final test predictions. Use `CONFIG_PATH` or `--config` to select another configuration and `WORKDIR` or `--workdir` for a custom output directory.

The [main README](../../README.md) documents setup, environment variables and the UniRank reference-run protocol.

## Numerical audit

After setup, point the audit to a separate UniRank source checkout:

```sh
./scripts/check_parity.sh --unirank-root /path/to/UniRank
```

The installed Perseus-CTR library and this repository's baseline configs are found automatically. No dataset download or training is required. The checks run on CPU with aligned inputs, parameters, padding conventions and context-field order. They cover outputs, a common loss, input gradients and event-projection gradients from embeddings to predictions for EST, HyFormer, MixFormer and HeMix on both datasets. TokenMixer and HiFormer are checked on their corresponding internal blocks, comparing outputs and input gradients.

Outputs use `atol=1e-6, rtol=1e-5`; input gradients use `atol=1e-6, rtol=1e-4`; event-projection gradient absolute errors must be below `1e-6`. Reports are saved under `benchmark/audit/`. A failed numerical check exits with a nonzero status. These checks do not establish equivalence of preprocessing, embedding-table learning or complete training runs.

## UniRank reference runs

[UniRank](https://github.com/salmon1802/UniRank) is run from a separate checkout and environment against the same blocked datasets. Follow its installation instructions and set the local dataset paths in `config/dataset_config.yaml`, including the train/valid/test user and item information paths.

```sh
git clone https://github.com/salmon1802/UniRank.git
cd UniRank
```

Our local reference setup used one GPU and one process, BF16, one epoch, microbatch 8192, four accumulation steps (effective batch 32768), four loader workers and `drop_last: false`. Dense parameters used AdamW with learning rate `1e-4`; embeddings used Adagrad with learning rate `0.05`. Architecture-specific settings were retained from the corresponding UniRank experiment definitions. Repeated runs used seeds 42–54.

Before launching, update the following fields in the existing `config/model_config.yaml`; retain its other model settings:

```yaml
Base:
  num_workers: 4
  drop_last: false
EST_Taobao_Action:
  seed: 42
  epochs: 1
  batch_size: 8192
  accumulation_steps: 4
```

Then run the native entry point:

```sh
python run_expid.py --config ./config --expid EST_Taobao_Action --gpu 0
```

For MerRec use `EST_MerRec_Action` and update the same fields in that experiment. Replace `EST` with `HyFormer`, `MixFormer`, `HeMix`, `HiFormer` or `TokenMixer` for other architectures. Set a separate output location for each seed through the upstream configuration. TokenMixer reference retraining was unsuccessful in our experiments.
