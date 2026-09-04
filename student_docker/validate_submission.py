"""제출 전에 반드시 실행하세요.

채점 서버가 사용하는 것과 정확히 동일한 아키텍처와 동일한 strict=True
load_state_dict 호출로 체크포인트를 로드한 뒤, 실제 forward를 한 번
수행합니다. 이 스크립트가 통과하면 채점 서버가 로드 단계에서 체크포인트를
거부하는 일은 없습니다. shape 불일치나 저장 코드의 오타는 제출 기회를
소모하지 않고 여기서 미리 걸러집니다.

    python validate_submission.py --ckpt models/experiment-001.pt
"""
import argparse

import torch

from imagenet_vit import ViTWrapper


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--num_classes", type=int, default=100)
    args = p.parse_args()

    m = ViTWrapper(num_classes=args.num_classes, pretrained=False,
                   drop_path_rate=0.0, in_model_norm=False)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    m.load_state_dict(sd.get("model", sd), strict=True)
    m.eval()

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = m(x)
    assert logits.shape == (2, args.num_classes), \
        f"logits shape이 (2, {args.num_classes})여야 하는데 {tuple(logits.shape)}입니다"

    print(f"통과 -- {args.ckpt} 가 정상적으로 로드됩니다 (strict=True). "
          f"forward 결과 logits shape: {tuple(logits.shape)}")
    print("이 체크포인트는 제출 가능한 상태입니다.")


if __name__ == "__main__":
    main()
