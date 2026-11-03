import argparse
import hashlib
import json
import sys
from pathlib import Path

import perseus

from perseus_lib.cross_domain._config import load_recipe, save_recipe
from perseus_lib.cross_domain._train import revision

from .train_lcn import main as train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--stores", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recipe = load_recipe(args.config, args.seed, lcn=True)
    root = Path(__file__).resolve().parents[2]
    library = revision(Path(perseus.__file__).resolve().parents[1])
    sys.argv = [
        "lcn",
        "--workdir",
        str(args.data),
        "--output",
        str(args.output),
    ]
    if args.stores is not None:
        sys.argv.extend(["--stores", str(args.stores)])
    for name, value in recipe.items():
        sys.argv.append("--" + name.replace("_", "-"))
        if isinstance(value, list):
            sys.argv.extend(str(v) for v in value)
        elif not isinstance(value, bool):
            sys.argv.append(str(value))
        elif not value:
            raise ValueError("Omit false flags")
    train()
    save_recipe(args.output / "effective-config.yaml", recipe)
    (args.output / "source.json").write_text(
        json.dumps(
            {
                "library_revision": library,
                "experiments_revision": revision(root),
                "source_config_sha256": hashlib.sha256(
                    args.config.read_bytes()
                ).hexdigest(),
                "config_sha256": hashlib.sha256(
                    (args.output / "effective-config.yaml").read_bytes()
                ).hexdigest(),
                "recipe": recipe,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
