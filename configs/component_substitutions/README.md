# Grid 2: component substitutions on Taobao

Configurations explore sequence encoding, non-sequential processing, interaction and readout changes for EST, HyFormer and HeMix. Each YAML specifies the full variant, including its training settings.

| Directory under `configs/taobao/` | Experiments |
|---|---|
| `ladder/` | Cumulative changes: positions, shared non-sequential FFN and readout/query variants |
| `block-transfer/` | Component substitutions across architectures |
| `position-shared-ffn/` | Sequence, non-sequential and interaction variants, including HeMix controls without sinusoidal positions |

Baselines are the corresponding [baseline Taobao configurations](../baseline_comparison/taobao/). Step numbers in `ladder/` describe each architecture's own sequence. Other branches are independent variants; their directory order does not define a sequence of changes.

## Prepare and train

Run from the repository root after [setup](../../README.md#setup):

```sh
./scripts/download.sh taobao
./scripts/data_preprocessing/taobao.sh
CONFIG_PATH=configs/component_substitutions/taobao/ladder/est-step2-shared-ffn.yaml \
  ./scripts/perseus.sh taobao est --seed 42
```

The launcher prepares data for the selected recipe and trains it. Keep the parent variant's data, training budget and effective batch fixed, and use matching seeds for controlled comparisons. Prepared-data compatibility is checked automatically.

## Selected configurations and direct parents

All paths below are relative to `configs/component_substitutions/taobao/`, except the Grid 1 baseline links. Other configurations are additional exploration variants.

| Architecture | Configuration | Direct parent | YAML |
|---|---|---|---|
| EST | Baseline | — | [Grid 1 baseline](../baseline_comparison/taobao/ctr-v1-taobao-reproduce-est.yaml) |
| EST | Sinusoidal positions | Baseline | [ladder/est-step1-position.yaml](taobao/ladder/est-step1-position.yaml) |
| EST | Shared non-sequential FFN | Sinusoidal positions | [ladder/est-step2-shared-ffn.yaml](taobao/ladder/est-step2-shared-ffn.yaml) |
| EST | ModernBERT encoder | Shared non-sequential FFN | [position-shared-ffn/est-s-no-rope.yaml](taobao/position-shared-ffn/est-s-no-rope.yaml) |
| EST | Rank-16 mixing | Shared non-sequential FFN | [position-shared-ffn/est-n.yaml](taobao/position-shared-ffn/est-n.yaml) |
| EST | Mean context readout | Shared non-sequential FFN | [ladder/est-step3-context-mean.yaml](taobao/ladder/est-step3-context-mean.yaml) |
| HyFormer | Baseline | — | [Grid 1 baseline](../baseline_comparison/taobao/ctr-v1-taobao-reproduce-hyformer.yaml) |
| HyFormer | Sinusoidal positions | Baseline | [ladder/hyformer-step1-position.yaml](taobao/ladder/hyformer-step1-position.yaml) |
| HyFormer | Shared non-sequential FFN | Sinusoidal positions | [ladder/hyformer-step2-shared-ffn.yaml](taobao/ladder/hyformer-step2-shared-ffn.yaml) |
| HyFormer | ModernBERT encoder | Shared non-sequential FFN | [position-shared-ffn/hyformer-s-no-rope.yaml](taobao/position-shared-ffn/hyformer-s-no-rope.yaml) |
| HyFormer | Rank-16 mixing | Shared non-sequential FFN | [position-shared-ffn/hyformer-n.yaml](taobao/position-shared-ffn/hyformer-n.yaml) |
| HeMix | Baseline | — | [Grid 1 baseline](../baseline_comparison/taobao/ctr-v1-taobao-reproduce-hemix.yaml) |
| HeMix | Sinusoidal positions | Baseline | [ladder/hemix-step1-position.yaml](taobao/ladder/hemix-step1-position.yaml) |
| HeMix | Shared non-sequential FFN | Sinusoidal positions | [ladder/hemix-step2-shared-ffn.yaml](taobao/ladder/hemix-step2-shared-ffn.yaml) |
| HeMix | ModernBERT encoder | Shared non-sequential FFN | [position-shared-ffn/hemix-s-no-rope.yaml](taobao/position-shared-ffn/hemix-s-no-rope.yaml) |
| HeMix | Rank-256 attention | Shared non-sequential FFN | [position-shared-ffn/hemix-n.yaml](taobao/position-shared-ffn/hemix-n.yaml) |

The ModernBERT variants use absolute positions with RoPE disabled. The non-sequential replacement uses rank-16 mixing for EST/HyFormer and rank-256 attention for HeMix.
