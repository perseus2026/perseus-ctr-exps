from pathlib import Path

import yaml


def load_recipe(path: Path, seed: int | None = None, *, lcn: bool = False) -> dict:
    recipe = yaml.safe_load(path.read_text())
    if seed is not None:
        if lcn:
            recipe["seed"] = seed
        else:
            recipe["training"]["seed"] = seed
    return recipe


def save_recipe(path: Path, recipe: dict) -> None:
    path.write_text(yaml.safe_dump(recipe, sort_keys=False))


# Paper study settings. Additional details are documented in configs/cross_domain/README.md.
STUDIES = {
    "tecd": {"seeds": tuple(range(42, 47)), "global_batch": 32768,
             "targets": ("click", "cart", "order")},
    "taobao_ads": {"seeds": tuple(range(42, 52)), "global_batch": 2048,
                   "targets": ("click",)},
}
