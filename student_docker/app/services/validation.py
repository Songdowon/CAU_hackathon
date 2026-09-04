"""Fast architecture check using the immutable official model definition.

Full public validation scoring is launched explicitly from score_model.py.
This lighter endpoint remains useful for catching a bad state_dict first. Participant
``.pt`` bytes are converted under the participant UID first; this trusted
process loads only safetensors.
"""
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors

from app.core import config
from app.services.checkpoint_conversion import convert_checkpoint
from imagenet_vit import ViTWrapper


def check_submission(ckpt_path: str, num_classes: int = 100) -> dict:
    with tempfile.TemporaryDirectory(prefix="hackathon-validation-") as temporary:
        converted = convert_checkpoint(
            Path(ckpt_path),
            Path(temporary) / "submission.safetensors",
            max_bytes=config.MAX_ARTIFACT_BYTES,
        )
        state = load_safetensors(str(converted), device="cpu")
        m = ViTWrapper(
            num_classes=num_classes,
            pretrained=False,
            drop_path_rate=0.0,
            in_model_norm=False,
        )
        m.load_state_dict(state, strict=True)
        m.eval()

        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = m(x)

    return {
        "status": "ok",
        "loads_cleanly": True,
        "logits_shape": list(logits.shape),
        "num_parameters": sum(p.numel() for p in m.parameters()),
        "message": "구조 검사를 통과했습니다. score_model.py로 공개 validation "
                   "점수를 확인하거나 이 모델을 비공개 test 채점에 제출할 수 있습니다.",
    }
