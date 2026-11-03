# Cross-domain training (Grid 3)

Five models are available: Perseus, EST, HeMix, HyFormer and an independently adapted LCN. Each model has one YAML per history setting, `single` or `cross`; the seed is supplied at launch.

| Dataset | Configurations | History protocol | Reference seeds |
|---|---|---|---|
| `taobao_ads` | `taobao_ads/<model>_<history>.yaml` | Single: 50 target events. Cross: the same target events plus up to 500 auxiliary events. | 42–51 |
| `tecd` | `tecd/<model>_<history>.yaml` | Single: Retail. Cross: Retail, Marketplace, Offers and Payments under a shared 512-event cap. | 42–46 |

## Download and preprocess

Run from the repository root after [setup](../../README.md#setup):

```sh
./scripts/download.sh taobao_ads
./scripts/data_preprocessing/taobao_ads.sh
# Or, for T-ECD:
./scripts/download.sh tecd
./scripts/data_preprocessing/tecd.sh
```

Download and preprocessing are separate commands. See [Taobao Ads](../../data_preprocessing/README.md#taobao-ads) and [T-ECD](../../data_preprocessing/README.md#t-ecd) for sources and preprocessing options.

## Train and evaluate

```sh
./scripts/perseus.sh taobao_ads hemix --history cross --seed 42
./scripts/perseus.sh tecd est --history single --seed 42
```

Available models are `perseus`, `est`, `hemix` and `hyformer`. Use the separate command for LCN:

```sh
./scripts/lcn.sh taobao_ads --history cross --seed 42
./scripts/lcn.sh tecd --history single --seed 42
```

Required domain-separated Taobao LCN stores are prepared automatically on first use. LCN is an independent adaptation, not a verified reproduction of author code.

Outputs go to `benchmark/<dataset>/runs/<model>-<history>-s<seed>/`. Perseus writes metrics to `reports/test-metrics.json`, predictions to `reports/test-predictions.pq` and its checkpoint to `checkpoint/`. LCN writes metrics to `summary.json`.

## Settings

Taobao Ads predicts observed impression clicks; categories below 2000 define the target domain. Histories contain only earlier events. T-ECD predicts recorded click/cart/order indicators over Retail events, using a chronological 80/20 split, a 12-hour history gap and a test-boundary embargo. Its shared history cap allows auxiliary events to displace target events.

Both studies use one fixed epoch without validation/checkpoint selection. Effective batches are 2048 for Taobao Ads and 32768 for T-ECD. The YAML records the training and hardware settings. The launchers check the study seed sets and batch/data compatibility; they do not require a Git checkout or fixed library revision.

Select another recipe with `CONFIG_PATH` or `--config`, and another output directory with `WORKDIR` or `--workdir`. For these studies, hardware settings come from the YAML rather than the Taobao/MerRec CLI overrides.
