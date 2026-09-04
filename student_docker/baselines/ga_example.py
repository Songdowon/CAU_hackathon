"""Minimal (and deliberately weak) working example of the required shape:
load M_o, touch the forget set, save a checkpoint.

This is NegGrad -- gradient ASCENT on the forget set only, nothing else. It
"forgets" by damaging the model on the forget classes and, since it never sees
retain data, also damages other classes along the way. Do not build your
submission by copying this and tuning lr/epochs: it exists to show the required
checkpoint shape, not as a competitive starting point.

    python baselines/ga_example.py --config configs/unlearn.yaml
"""
import argparse
import os

import torch
import torch.nn.functional as F
import yaml

from imagenet_vit import ViTWrapper
from utils.data import get_loaders


def load_mo(ckpt, num_classes, device):
    m = ViTWrapper(num_classes=num_classes, pretrained=False,
                   drop_path_rate=0.0, in_model_norm=False)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    m.load_state_dict(sd.get("model", sd), strict=True)
    return m.to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/unlearn.yaml")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--save_path", default="models/ga_example.pt")
    p.add_argument("--max_batches", type=int, default=0,
                   help="stop each epoch after this many batches; 0 = full "
                        "forget set. Use a small number for a fast sanity "
                        "check before committing to a full run.")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))

    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_mo(cfg["model"]["mo_ckpt"], cfg["model"]["num_classes"], device)
    loaders = get_loaders(model, batch_size=cfg["data"]["batch_size"],
                          workers=cfg["data"]["workers"], seed=cfg["seed"])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    model.train()
    for ep in range(args.epochs):
        tot = n = 0
        for x, y in loaders["forget"]:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                loss = -F.cross_entropy(model(x), y)  # ascent: maximize forget loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss)
            n += 1
            if args.max_batches and n >= args.max_batches:
                break
        print(f"[ga] epoch {ep + 1}/{args.epochs} loss {tot / max(n, 1):.4f} "
              f"({n} batches)")

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save({"model": model.state_dict()}, args.save_path)
    print(f"saved {args.save_path}")
    print(f"structure check: python validate_submission.py --ckpt {args.save_path}")
    print(f"local validation: python score_model.py {args.save_path}")


if __name__ == "__main__":
    main()
