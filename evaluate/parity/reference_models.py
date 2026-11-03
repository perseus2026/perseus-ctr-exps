"""Local architecture audit: source/config reads and synthetic CPU/meta modules only."""
from pathlib import Path
import ast
import csv
import gc
import json
import sys
import os
import types
from collections import OrderedDict

sys.dont_write_bytecode = True
ROOT = Path(os.environ["PERSEUS_ROOT"]).resolve()
P = ROOT
U = Path(os.environ['UNIRANK_ROOT']).resolve()
OUT = Path(os.environ.get('PARITY_OUTPUT_DIR', 'benchmark/audit'))
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(P))
import torch
from torch import nn
import yaml
from perseus.core.config import Config
from perseus.core.backbone import Backbone
from perseus.core.tasks.classification.target import Layer, Preprocessor

torch.set_num_threads(1)
torch.manual_seed(20261025)
# Load the reference components without importing its training infrastructure.
import load_reference
uns = vars(load_reference)

class NoInfrastructure(nn.Module):
    """No training, device movement, reset, files or optimizer side effects."""
    def __init__(self, *args, **kwargs):
        super().__init__()
    def compile(self, *args, **kwargs):
        pass
    def reset_parameters(self):
        pass
    def model_to_device(self):
        pass
    def get_output_activation(self, *args):
        return nn.Sigmoid()
    def activation_checkpoint(self, module, *args):
        return module(*args)
    def get_inputs(self, inputs):
        return inputs

class EmbeddingsAlreadySupplied(nn.Module):
    """Synthetic floating embeddings are supplied directly; no vocabulary tables."""
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, features, flatten_emb=False):
        x = torch.stack(list(features.values()), dim=-2)
        return x.flatten(start_dim=-2) if flatten_emb else x

NAMES = {'est':'EST', 'hyformer':'HyFormer', 'mixformer':'MixFormer',
         'tokenmixer':'TokenMixer', 'hiformer':'HiFormer', 'hemix':'HeMix'}
UMODULES = {}
for key, name in NAMES.items():
    mod = uns[{'est':'est_mod','hyformer':'hyformer_mod','mixformer':'mixformer_mod',
               'tokenmixer':'tokenmixer_mod','hiformer':'hiformer_mod','hemix':'hemix_mod'}[key]]
    mod.MultiTaskModel = NoInfrastructure
    mod.FeatureEmbedding = EmbeddingsAlreadySupplied
    source = U / 'model_zoo' / (name + '.py')
    node = next(x for x in ast.parse(source.read_text()).body if isinstance(x, ast.ClassDef) and x.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), mod.__dict__)
    UMODULES[key] = mod

def pconfig(dataset, family):
    config_root = Path(__file__).resolve().parents[2] / 'configs/baseline_comparison'
    path = config_root / dataset / f'ctr-v1-{dataset}-reproduce-{family}.yaml'
    return path, Config.model_validate(yaml.safe_load(path.read_text()))

def pmodel(c, dataset):
    event_fields = c.feature_to_encoder(event=True)
    context_fields = c.feature_to_encoder(context=True)
    b = Backbone(c.backbone.dim, embedding_dim=c.embedding_dim,
        event_aggregator_config=dict(num_features=len(event_fields), **c.backbone.event_aggregator.model_dump()),
        context_aggregator_config=dict(num_features=len(context_fields), **c.backbone.context_aggregator.model_dump()),
        history_aggregator_config=dict(max_events_per_sequence=c.max_events_per_sequence, **c.backbone.history_aggregator.model_dump()),
        context_fields=list(context_fields), readout_token=c.backbone.readout_token,
        mask_context_from_history=c.backbone.mask_context_from_history)
    head = Layer.init(Preprocessor({name: ['0', '1'] for name in fmap(dataset).labels}),
                      b.output_shape[-1], **c.task.layer)
    return b, head

MC = yaml.safe_load((U/'config/model_config.yaml').read_text())
DC = yaml.safe_load((U/'config/dataset_config.yaml').read_text())

def uconfig(dataset, family):
    return MC['Base'] | MC[NAMES[family] + '_' + {'taobao':'Taobao','merrec':'MerRec'}[dataset] + '_Action']

def fmap(dataset):
    d = DC[{'taobao':'Taobao','merrec':'MerRec'}[dataset] + '_Action']
    features = OrderedDict((x['name'], x) for x in d['feature_cols'] if x['type'] != 'meta' and x.get('active',True))
    return types.SimpleNamespace(features=features, labels=[x['name'] for x in d['label_col']],
                                 get_num_fields=lambda: len(features))

def umodel(dataset, family):
    return getattr(UMODULES[family], NAMES[family])(fmap(dataset), **uconfig(dataset,family))

