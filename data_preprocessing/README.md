# Data preparation

Run from the repository root after [setup](../README.md#setup). First download the dataset, then preprocess the local source files.

| Dataset | Source | Download | Preprocess |
|---|---|---|---|
| Taobao actions | [UniRank Taobao](https://huggingface.co/datasets/salmon1802/Taobao) | `./scripts/download.sh taobao` | `./scripts/data_preprocessing/taobao.sh` |
| MerRec actions | [UniRank MerRec](https://huggingface.co/datasets/salmon1802/MerRec) | `./scripts/download.sh merrec` | `./scripts/data_preprocessing/merrec.sh` |
| Taobao Ads | [HF tables](https://huggingface.co/datasets/Angrybird12/ad-display_click-data_taobao.com), [Kaggle histories](https://www.kaggle.com/datasets/zerotrustai/taobao-ad-display-click) | `./scripts/download.sh taobao_ads` | `./scripts/data_preprocessing/taobao_ads.sh` |
| T-ECD | [Hugging Face](https://huggingface.co/datasets/t-tech/T-ECD) | `./scripts/download.sh tecd` | `./scripts/data_preprocessing/tecd.sh` |

Download collects and validates source files. The existing Taobao Ads collector unpacks/caches history as Parquet; the T-ECD collector creates Small-filtered Payments caches. Neither stage creates training checkpoints. Preprocessing then builds the event/sample representation and, for cross-domain studies, the model-ready datasets. It performs no network downloads.

## Paths and options

Downloaded sources go to `benchmark/<dataset>/raw/`. Taobao/MerRec conversion produces `benchmark/<dataset>/converted/` containing `hub/` and `basis/`. Model-specific preparation is performed automatically by `scripts/perseus.sh` and stored under `benchmark/<dataset>/prepared/<configuration>/`.

Cross-domain preprocessing creates `benchmark/<dataset>/prepared/{single,cross}/`, each containing `checkpoint/` and `dataset/`.

```sh
# Preprocess an existing download.
./scripts/data_preprocessing/taobao.sh

# Supply explicit source and output locations.
./scripts/data_preprocessing/merrec.sh \
  --raw-root /data/merrec-blocked --out-root /data/merrec-converted
```

Set the same converted location with `DATA_ROOT=/data/merrec-converted` when training. `RAW_ROOT`, `DATA_ROOT` and `BENCHMARK_ROOT` can also be supplied as environment variables. Use `--dry-run` to print the steps. Additional converter options are forwarded; inspect them through `python -m data_preprocessing.<dataset>.convert --help`.

## Dataset protocols

### UniRank Taobao and MerRec

Taobao/MerRec use the original blocked files from the UniRank Hugging Face releases, including metadata and split-specific user/item information. The automatically generated dataset-viewer exports are not substitutes. Conversion preserves source keys and validation/test tags, and sets candidate action to zero. Rebuilding UniRank's blocked releases from original events is documented in [UniRank](https://github.com/salmon1802/UniRank).

### Taobao Ads

Sources are the [Hugging Face impression/feature tables](https://huggingface.co/datasets/Angrybird12/ad-display_click-data_taobao.com), [Kaggle behaviour histories](https://www.kaggle.com/datasets/zerotrustai/taobao-ad-display-click) and the [original dataset page](https://tianchi.aliyun.com/dataset/56). This is a separate impression-level protocol from UniRank Taobao actions.

The split is May 7–12 train / May 13 test, 2017, Asia/Shanghai. Labels are observed `clk`, without sampled negative items. Categories below 2000 define the target domain. Single retains 50 target events; Cross keeps those same events and adds up to 500 auxiliary events. Histories contain only earlier events.

Preprocessing writes `benchmark/taobao_ads/prepared/{single,cross}/`, with `checkpoint/` and `dataset/` under each setting. Use `--scratch` to select temporary storage. If the sources are already present, run only preprocessing.

### T-ECD

The [T-ECD source](https://huggingface.co/datasets/t-tech/T-ECD) provides Small Retail, Marketplace and Offers, plus Full Payments filtered to Small users and brands. Full Payments requires about 887 GiB of source transfer.

The task predicts recorded click/cart/order indicators on Retail events; views have zero labels. The split is chronological 80/20, with a 12-hour history gap and a test-boundary embargo. Both settings use a 512-event total cap: Single includes Retail only, while Cross also includes Marketplace, Offers and Payments.

Preprocessing writes `benchmark/tecd/prepared/{single,cross}/`, with `checkpoint/` and `dataset/` under each setting. `--payments` accepts a separate collected Payments source root; `--scratch` selects temporary storage. Run only preprocessing to reuse collected sources.

Continue with the [cross-domain training commands](../configs/cross_domain/README.md).

## Prepared-data compatibility

Prepared data must match the selected configuration's fields, feature order, encoders and history settings. The launchers check compatibility before training. Cached readers keep events in each process's memory; allocate sufficient host RAM.
