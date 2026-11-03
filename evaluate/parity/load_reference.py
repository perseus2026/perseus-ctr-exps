import os
import importlib.util
import sys
import types
from pathlib import Path

import torch

UNIRANK = Path(os.environ["UNIRANK_ROOT"])


def load(name: str, relpath: str, pkg: types.ModuleType) -> types.ModuleType:
    module = types.ModuleType(name)
    if relpath.endswith("__init__.py"):
        module.__package__ = name
        module.__path__ = [str((UNIRANK / relpath).parent)]
    else:
        module.__package__ = pkg.__name__
    sys.modules[name] = module
    spec = importlib.util.spec_from_file_location(name, UNIRANK / relpath)
    spec.loader.exec_module(module)
    return module


unirank = types.ModuleType("unirank")
unirank.__path__ = [str(UNIRANK)]
sys.modules["unirank"] = unirank
pytorch = types.ModuleType("unirank.pytorch")
pytorch.__path__ = [str(UNIRANK / "unirank" / "pytorch")]
sys.modules["unirank.pytorch"] = pytorch

torch_utils = load("unirank.pytorch.torch_utils", "unirank/pytorch/torch_utils.py", pytorch)
layers = types.ModuleType("unirank.pytorch.layers")
layers.__path__ = [str(UNIRANK / "unirank" / "pytorch" / "layers")]
sys.modules["unirank.pytorch.layers"] = layers
models = types.ModuleType("unirank.pytorch.models")
models.MultiTaskModel = object
sys.modules["unirank.pytorch.models"] = models
utils = types.ModuleType("unirank.utils")
utils.not_in_whitelist = lambda x: x
sys.modules["unirank.utils"] = utils

blocks_mod = load("unirank.pytorch.layers.blocks", "unirank/pytorch/layers/blocks/__init__.py", layers)
interactions = load("unirank.pytorch.layers.interactions", "unirank/pytorch/layers/interactions/__init__.py", layers)
tokenization = load("unirank.pytorch.layers.tokenization", "unirank/pytorch/layers/tokenization/__init__.py", layers)
attentions = load("unirank.pytorch.layers.attentions", "unirank/pytorch/layers/attentions/__init__.py", layers)
pooling = load("unirank.pytorch.layers.pooling", "unirank/pytorch/layers/pooling.py", layers)
for module in (blocks_mod, interactions, tokenization, attentions, pooling):
    for attr in dir(module):
        if not attr.startswith("_"):
            setattr(layers, attr, getattr(module, attr))
layers.FeatureEmbedding = object
layers.FeatureEmbeddingDict = object

model_zoo = types.ModuleType("model_zoo")
model_zoo.__path__ = [str(UNIRANK / "model_zoo")]
sys.modules["model_zoo"] = model_zoo
tokenmixer_mod = load("model_zoo.TokenMixer", "model_zoo/TokenMixer.py", model_zoo)
hiformer_mod = load("model_zoo.HiFormer", "model_zoo/HiFormer.py", model_zoo)
hemix_mod = load("model_zoo.HeMix", "model_zoo/HeMix.py", model_zoo)
est_mod = load("model_zoo.EST", "model_zoo/EST.py", model_zoo)
hyformer_mod = load("model_zoo.HyFormer", "model_zoo/HyFormer.py", model_zoo)
mixformer_mod = load("model_zoo.MixFormer", "model_zoo/MixFormer.py", model_zoo)

