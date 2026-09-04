"""sweep용 config 생성기.

    python tools/mkcfg.py r002 "설명" train.steps=1200 train.lr=3e-5

configs/r001.yaml을 베이스로 점 표기 키를 덮어써서 configs/<name>.yaml을 만들고,
첫 줄에 설명 주석을, output.save_path에 models/<name>.pt를 넣는다.
"""
import sys
from pathlib import Path

import yaml

BASE = Path("configs/r001.yaml")


def cast(v):
    for fn in (int, float):
        try:
            return fn(v)
        except ValueError:
            pass
    return {"true": True, "false": False, "none": None}.get(v.lower(), v)


def main():
    name, note, *overrides = sys.argv[1:]
    cfg = yaml.safe_load(BASE.read_text())
    for ov in overrides:
        key, val = ov.split("=", 1)
        node = cfg
        *path, leaf = key.split(".")
        for k in path:
            node = node.setdefault(k, {})
        node[leaf] = cast(val)
    cfg["output"]["save_path"] = f"models/{name}.pt"
    out = Path(f"configs/{name}.yaml")
    out.write_text(f"# {name} — {note}\n" + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print(out)


if __name__ == "__main__":
    main()
