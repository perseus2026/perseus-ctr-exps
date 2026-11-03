# Perseus-CTR experiments

Companion experiment repository for the paper **Perseus-CTR: Composable Sequence and Feature Interaction for Recommendation Research**.

This repository provides the dataset download/preprocessing pipelines, configuration files and training commands for the paper's three experimental studies. It uses the Perseus-CTR library and an external [UniRank](https://github.com/salmon1802/UniRank) checkout for reference runs.

## Experiments in the paper

| Study | Research question | Configurations and instructions |
|---|---|---|
| **Grid 1 — Architecture reproduction** | How do six Perseus-CTR recipes compare with the corresponding UniRank implementations on Taobao and MerRec? | [Baseline comparison](configs/baseline_comparison/README.md): 12 configs |
| **Grid 2 — Component substitutions** | What changes when sequence encoding, non-sequential processing, interaction or readout is replaced within EST, HyFormer and HeMix? | [Component substitutions](configs/component_substitutions/README.md): 43 configs, plus Grid 1 baselines |
| **Grid 3 — Cross-domain evaluation** | How do the same architectures respond to auxiliary-domain histories, compared with joint-sequence Perseus and an adapted LCN? | [Cross-domain training](configs/cross_domain/README.md): 20 configs |

Grid 3 contains **two studies**, each with Single and Cross history settings. Taobao Ads predicts observed impression clicks: Single uses 50 target-domain events, while Cross keeps those events and adds up to 500 auxiliary events. T-ECD predicts recorded Retail click/cart/order actions with a shared 512-event history cap; Cross adds Marketplace, Offers and Payments. Five models are supplied for each study: Perseus, EST, HeMix, HyFormer and LCN.

The YAMLs define the released recipes; training seeds are supplied through the CLI. The commands below download/preprocess data and run training with test evaluation. The [numerical audit](configs/baseline_comparison/README.md#numerical-audit) compares the tested computations with an external UniRank checkout. Seed-level statistical analysis and manuscript assets are outside the scope of this training repository.

## Layout

```text
data_preprocessing/       Dataset download and preprocessing
configs/                  Configurations grouped by experiment
  baseline_comparison/    Six architectures on Taobao and MerRec
  component_substitutions/ Sequence, feature-processing and interaction variants
  cross_domain/           Single- and cross-domain recipes
scripts/                  User-facing preprocessing and training commands
perseus_lib/              Data preparation and Perseus training implementation
baselines/lcn/            Adapted LCN baseline
benchmark/                Downloaded data and run outputs, created locally
```

## Datasets

| Dataset | Source | Experiments |
|---|---|---|
| `taobao` | [UniRank Taobao](https://huggingface.co/datasets/salmon1802/Taobao) | Baseline comparison and component substitutions |
| `merrec` | [UniRank MerRec](https://huggingface.co/datasets/salmon1802/MerRec) | Baseline comparison |
| `taobao_ads` | [HF impression tables](https://huggingface.co/datasets/Angrybird12/ad-display_click-data_taobao.com), [Kaggle behaviour histories](https://www.kaggle.com/datasets/zerotrustai/taobao-ad-display-click) | Cross-domain training |
| `tecd` | [T-ECD](https://huggingface.co/datasets/t-tech/T-ECD) | Cross-domain training |

The download command collects the required source files; preprocessing is a separate step. Taobao and MerRec use UniRank's blocked releases; `taobao_ads` is a separate impression-level preparation protocol. T-ECD includes Full Payments, requiring about 887 GiB of source transfer. See [data preparation](data_preprocessing/README.md) for details.

## Setup

Requires Python 3.12 and a local Perseus-CTR checkout.[^platforms] For sibling GitHub clones named `perseus-ctr` and `perseus-ctr-exps`, run from `perseus-ctr-exps`:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
./scripts/setup.sh ../perseus-ctr
```

For the unpacked release ZIPs (`repo/` and `exps-for-publication/`), use `./scripts/setup.sh ../repo` instead. Pass another library path if needed. GPU training requires a CUDA-compatible PyTorch installation. UniRank uses a separate environment.

[^platforms]: The documented local workflows use Linux/macOS; CUDA training uses Linux. On Windows, run the benchmarks under WSL2 with repositories and data in the Linux filesystem (for example, `~/...`, not `/mnt/c/...`).

## Running the paper experiments

The workflow has three stages: **download → preprocess → train and evaluate**. Model-specific preparation is performed automatically before training. All generated files are stored under `benchmark/<dataset>/`.

### 1. Download

```sh
./scripts/download.sh taobao
```

This stage collects source files only; it does not build model checkpoints or training datasets. Existing downloads are reused by the source clients. Taobao Ads collection also unpacks and converts the history archive to a local columnar cache; T-ECD collection filters Payments to the Small user/brand population. These source caches are inputs to the next stage.

| Dataset argument | Download source | Local source directory |
|---|---|---|
| `taobao` | UniRank Taobao blocked release | `benchmark/taobao/raw/` |
| `merrec` | UniRank MerRec blocked release | `benchmark/merrec/raw/` |
| `taobao_ads` | HF impression tables and Kaggle behaviour archive | `benchmark/taobao_ads/raw/` |
| `tecd` | T-ECD Small domains and Full Payments | `benchmark/tecd/raw/` |

### 2. Preprocess

```sh
./scripts/data_preprocessing/taobao.sh
```

Preprocessing reads downloaded local files; it does not download data. Run it once per dataset/protocol, then reuse its outputs across matching model configurations.

| Command | Output | Next step |
|---|---|---|
| `./scripts/data_preprocessing/taobao.sh` | `benchmark/taobao/converted/` | Grid 1 or Grid 2 |
| `./scripts/data_preprocessing/merrec.sh` | `benchmark/merrec/converted/` | Grid 1 |
| `./scripts/data_preprocessing/taobao_ads.sh` | `benchmark/taobao_ads/prepared/{single,cross}/` | Grid 3 |
| `./scripts/data_preprocessing/tecd.sh` | `benchmark/tecd/prepared/{single,cross}/` | Grid 3 |

For custom locations, pass `--raw-root` and `--out-root`, or set `RAW_ROOT` and `DATA_ROOT`. See [data preparation](data_preprocessing/README.md) for source formats and preprocessing options.

### 3. Train and evaluate

Choose the study below. Each command runs one configuration and seed. Use a fresh run directory for each launch; additional seeds are supplied through `--seed`, not separate YAML copies.

#### Grid 1 — Architecture reproduction

```sh
./scripts/perseus.sh taobao est --seed 42
./scripts/perseus.sh merrec hyformer --seed 42
```

| Option | Available values |
|---|---|
| Dataset | `taobao`, `merrec` |
| Model | `est`, `hyformer`, `mixformer`, `tokenmixer`, `hiformer`, `hemix` |
| Configuration | `configs/baseline_comparison/<dataset>/ctr-v1-<dataset>-reproduce-<model>.yaml` |
| Reference training seeds | 42–54; select one using `--seed` |

Train the required dataset/model/seed combinations using the same command pattern. The launcher prepares matching model data and preserves the recipe's training settings. [Grid 1 instructions](configs/baseline_comparison/README.md) include the **[UniRank reference setup](configs/baseline_comparison/README.md#unirank-reference-runs)** for the other side of the comparison.

#### Grid 2 — Component substitutions

Reuse the preprocessed `taobao` dataset and select a component recipe with `CONFIG_PATH`:

```sh
CONFIG_PATH=configs/component_substitutions/taobao/ladder/est-step2-shared-ffn.yaml \
  ./scripts/perseus.sh taobao est --seed 42
```

| Configuration shown in the paper | Direct parent | Recipe under `configs/component_substitutions/taobao/` | Architectures |
|---|---|---|---|
| Baseline | — | The corresponding Grid 1 Taobao YAML | EST, HyFormer, HeMix |
| Sinusoidal positions | Baseline | `ladder/<model>-step1-position.yaml` | EST, HyFormer, HeMix |
| Shared non-sequential FFN | Sinusoidal positions | `ladder/<model>-step2-shared-ffn.yaml` | EST, HyFormer, HeMix |
| ModernBERT encoder | Shared non-sequential FFN | `position-shared-ffn/<model>-s-no-rope.yaml` | EST, HyFormer, HeMix |
| Non-sequential mixing replacement | Shared non-sequential FFN | `position-shared-ffn/<model>-n.yaml` | EST, HyFormer, HeMix |
| Mean context readout | Shared non-sequential FFN | `ladder/est-step3-context-mean.yaml` | EST only |

Use lowercase model names in paths. The ModernBERT variants use absolute positions with RoPE disabled. The mixing replacement is rank-16 mixing for EST/HyFormer and rank-256 attention for HeMix. Later substitutions are independent branches from the shared-FFN recipe, not cumulative with one another.

For each selected configuration, repeat with seeds 42–54 to obtain 13 paired runs. All 16 selected architecture/configuration combinations and their exact parent/YAML links are listed in the [Grid 2 instructions](configs/component_substitutions/README.md). Other supplied recipes are exploration variants outside that selected table.

#### Grid 3 — Cross-domain evaluation

```sh
./scripts/perseus.sh taobao_ads hemix --history single --seed 42
./scripts/perseus.sh taobao_ads hemix --history cross --seed 42
./scripts/perseus.sh tecd est --history single --seed 42
./scripts/perseus.sh tecd est --history cross --seed 42

# Adapted LCN baseline uses its own entry point.
./scripts/lcn.sh taobao_ads --history cross --seed 42
./scripts/lcn.sh tecd --history single --seed 42
```

| Option | Taobao Ads | T-ECD |
|---|---|---|
| Dataset argument | `taobao_ads` | `tecd` |
| Perseus recipes | `perseus`, `est`, `hemix`, `hyformer` | `perseus`, `est`, `hemix`, `hyformer` |
| Additional baseline | `scripts/lcn.sh` | `scripts/lcn.sh` |
| History settings | `single`, `cross` | `single`, `cross` |
| Reference training seeds | 42–51 | 42–46 |
| YAML directory | `configs/cross_domain/taobao_ads/` | `configs/cross_domain/tecd/` |

For the full study, run both history settings for every model and seed in the corresponding column. Required Taobao LCN domain stores are prepared automatically on first use. Cross-domain hardware and batch settings come from the YAML. The [Grid 3 instructions](configs/cross_domain/README.md) describe history budgets and the adapted LCN implementation.

## Outputs and options

| Run type | Metric output | Other outputs |
|---|---|---|
| Grid 1/2 | `<run>/train.log` | Checkpoint, resolved configuration, `predictions/test/` |
| Grid 3 Perseus recipes | `<run>/reports/test-metrics.json` | `reports/test-predictions.pq`, checkpoint |
| Grid 3 LCN | `<run>/summary.json` | Effective configuration and model outputs |

Run directories default to `benchmark/<dataset>/runs/<configuration>-s<seed>/`; cross-domain names use `<model>-<history>-s<seed>`. Command options override environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `CONFIG_PATH` | Selected baseline/model YAML | Experiment configuration |
| `BENCHMARK_ROOT` | `benchmark` | Root for local data and runs |
| `RAW_ROOT` | `benchmark/<dataset>/raw` | Downloaded sources |
| `DATA_ROOT` | `benchmark/<dataset>/converted` or `prepared` | Preprocessing output and training input |
| `WORKDIR` | Default run directory above | One training run |
| `SEED` | `42` | Default training seed |

For Taobao/MerRec, `--processes`, `--microbatch` and `--workers` control hardware use. Effective batch is `microbatch × processes × accumulation`; the launcher derives accumulation to preserve the recipe's effective batch. Worker count is per process. Cross-domain runs use their YAML settings instead.

Download, preprocessing and training commands accept `--help` and `--dry-run`. A dry run prints the planned commands without downloading data or starting training; it is not a full dataset or numerical-equivalence check.
