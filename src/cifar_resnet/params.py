from __future__ import annotations

import argparse

from torch import nn

from cifar_resnet.config import load_config
from cifar_resnet.models import build_model, count_parameters
from cifar_resnet.rules import COMPETITION_MAX_PARAMS


def layer_table(model: nn.Module) -> str:
    rows = [("stem", count_parameters(model.stem))]
    for i, stage in enumerate(model.stages):
        rows.append((f"stage{i + 1} ({len(stage)} blocks)", count_parameters(stage)))
    rows.append(("fc", count_parameters(model.fc)))
    total = count_parameters(model)
    width = max(len(name) for name, _ in rows) + 2
    lines = [f"{name:<{width}}{n:>12,}  {100 * n / total:5.1f}%" for name, n in rows]
    lines.append(f"{'total':<{width}}{total:>12,}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--summary", action="store_true", help="also print the layer breakdown")
    args = parser.parse_args(argv)

    for path in args.configs:
        cfg = load_config(path)
        model = build_model(cfg.model, enforce_budget=False)
        n = count_parameters(model)
        if n <= COMPETITION_MAX_PARAMS:
            status = "OK"
        elif n <= cfg.model.max_params:
            status = "TEACHER ONLY: over the 5M limit, cannot be submitted"
        else:
            status = "OVER BUDGET"
        print(f"{path:<45} {n:>12,} params  [{status}]")
        if args.summary:
            print(layer_table(model), end="\n\n")


if __name__ == "__main__":
    main()
