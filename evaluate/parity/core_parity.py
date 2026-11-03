"""Compare only core TokenMixer/HiFormer blocks with matched UniRank weights.

This deliberately excludes target attention, tokenization, full-model parity,
padding histories, prediction heads and training/data protocols.
"""
import argparse
import ast
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.runtime))
    from perseus.core.backbone.history.nonsequence import attention as attention_module
    from perseus.core.backbone.history.nonsequence import ffn as ffn_module
    from perseus.core.backbone.history.nonsequence import token_mixer
    assert Path(token_mixer.__file__).resolve().is_relative_to(args.runtime.resolve())
    namespace = {"torch": torch, "nn": nn, "F": F}

    def reference_classes(relative, names):
        path = args.reference / relative
        tree = ast.parse(path.read_text(), filename=str(path))
        selected = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
        assert {node.name for node in selected} == set(names)
        # Execute exact class ASTs, bypassing unrelated optional UniRank imports.
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)

    reference_classes("unirank/pytorch/layers/interactions/per_token_ffn.py", ["PerTokenFeedForward", "PerTokenSwiGLU"])
    reference_classes("unirank/pytorch/layers/interactions/token_mix.py", ["MultiHeadTokenMixing"])
    reference_classes("unirank/pytorch/layers/attentions/dot_product_attention.py", ["ScaledDotProductAttention"])
    reference_classes("model_zoo/TokenMixer.py", ["TokenMixerBlocks"])
    reference_classes("model_zoo/HiFormer.py", ["HiFormerBlocks", "LowRankLinear", "HiformerAttentionLayer"])
    torch.set_num_threads(1)
    torch.manual_seed(9)

    def copy_params(target, source, pairs):
        with torch.no_grad():
            for target_name, source_name in pairs:
                getattr(target, target_name).copy_(getattr(source, source_name))

    def compare(reference, modules, shape):
        first = torch.randn(shape, requires_grad=True)
        second = first.detach().clone().requires_grad_(True)
        expected = reference(first)
        actual = second
        for module in modules:
            actual = module(actual, None)
        expected.square().sum().backward()
        actual.square().sum().backward()
        result = {
            "input_shape": list(shape), "output_shape": list(expected.shape),
            "max_abs_output_error": float((expected - actual).abs().max().detach()),
            "max_abs_input_gradient_error": float((first.grad - second.grad).abs().max()),
        }
        assert torch.equal(expected, actual), result
        assert torch.equal(first.grad, second.grad), result
        return result

    reference = namespace["TokenMixerBlocks"](input_dim=8, num_tokens=5, num_layers=2, expand=2)
    modules = []
    dim = 8
    glu_pairs = [("w1", "W1"), ("w2", "W2"), ("w3", "W3"), ("b1", "b1"), ("b2", "b2")]
    for layer in range(2):
        mixing = token_mixer.ReversibleTokenMixing(dim, 5, 2, 0.0)
        dim = mixing.output_dim
        ffn = ffn_module.per_token_ffn(dim, 5, expansion_factor=2, activation="swiglu")
        mixing.mixer_norm.load_state_dict(reference.mixer_norms[layer].state_dict())
        ffn.norm.load_state_dict(reference.pffn_norms[layer].state_dict())
        copy_params(mixing.pffn, reference.pffn_layers1[layer], glu_pairs)
        copy_params(ffn.op, reference.pffn_layers2[layer], glu_pairs)
        modules.extend([mixing, ffn])
    tokenmixer = compare(reference, modules, (2, 5, 8))

    reference = namespace["HiFormerBlocks"](token_dim=8, num_token=6, num_layers=2, expand=2, num_heads=2, qkv_rank=4)
    modules = []
    for layer in range(2):
        attention = attention_module.low_rank_attention(8, 6, num_heads=2, qkv_rank=4)
        ffn = ffn_module.per_token_ffn(8, 6, expansion_factor=2, activation="gelu")
        attention.norm.load_state_dict(reference.mixer_norms[layer].state_dict())
        ffn.norm.load_state_dict(reference.pffn_norms[layer].state_dict())
        with torch.no_grad():
            for target_name, source_name in [("query", "W_q"), ("key", "W_k"), ("value", "W_v")]:
                target = getattr(attention.op, target_name)
                source = getattr(reference.mixer_layers[layer], source_name)
                target[0].weight.copy_(source.down_proj.weight)
                target[1].weight.copy_(source.up_proj.weight)
        copy_params(ffn.op, reference.pffn_layers[layer], [
            ("first_weight", "W1"), ("second_weight", "W2"), ("first_bias", "b1"), ("second_bias", "b2"),
        ])
        modules.extend([attention, ffn])
    hiformer = compare(reference, modules, (2, 6, 8))
    report = {
        "scope": "core blocks only; NOT full-model, tokenizer, target-attention, padding, heads or training parity",
        "device": "cpu", "dtype": "float32", "seed": 9, "torch_version": torch.__version__,
        "reference_class_loading": "unmodified class AST from the recorded source files",
        "tokenmixer": tokenmixer, "hiformer": hiformer,
        "status": "PASS",
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
