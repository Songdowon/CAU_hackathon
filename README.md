# ImageNet-100 Unlearning 학생 코드 백업

이 저장소에는 학생 코드와 public validation scorer만 있습니다. 데이터셋, 시작
모델, representation cache, 제출 checkpoint와 Hugging Face cache는 포함하지
않습니다.

학생 이미지와 동일한 Python/CUDA 환경이 준비된 서버를 기준으로 합니다.

## 서버에 설치

공유 스토리지에서 받은 archive를 home에 복사합니다.

```bash
mkdir -p ~/dataset
cp /공유/스토리지/경로/student_docker.tar ~/dataset/student_docker.tar
```

이 저장소를 home에 clone하고 설치 스크립트를 실행합니다.

```bash
cd ~
git clone https://github.com/사용자명/hackathon-student-code.git
cd ~/hackathon-student-code
./ops/setup-cloned-student-workspace.sh
```

스크립트는 `~/dataset/student_docker.tar`를 clone의 `student_docker/` 아래에 직접
풉니다. archive의 크기와 SHA-256, released/validation 이미지 수도 확인합니다.

작업을 시작할 때 환경을 활성화합니다.

```bash
cd ~/hackathon-student-code
source student_docker/activate_local.sh

python unlearn.py --config configs/unlearn.yaml
python validate_submission.py --ckpt models/experiment-001.pt
python score_model.py models/experiment-001.pt
```

archive가 기본 위치가 아니라면 경로를 직접 넘길 수 있습니다.

```bash
./ops/setup-cloned-student-workspace.sh /절대/경로/student_docker.tar
```

학생 과제 설명은 [student_docker/README.md](student_docker/README.md)를 참고하세요.
