# ⚠️ 이 디렉터리는 비어 있습니다 (1차 전달본)

학습용 ImageNet 이미지 **113,566장 / 13GB**가 들어갈 자리입니다.
용량 때문에 1차 전달본에서 제외했으며, **2차 전달본(구글 드라이브)** 에
`04_student_train_images.tar.part00~03` 으로 들어 있습니다.

## 현재 상태로 빌드하면 실패합니다

`Dockerfile`의 아래 구문이 `train/` 디렉터리를 찾지 못해 에러가 납니다.

```
COPY imagenet_released/train/ /data/hai_ssh/datasets/imagenet/train/
```

## 채우는 방법 (둘 중 하나)

### 방법 1 — 2차 전달본 받아서 풀기 (권장)

분할 파일이므로 **`cat`으로 이어붙여서** 풉니다.
푸는 위치는 `student_docker/`가 아니라 그 **상위 디렉터리(`hackathon/`)** 입니다.

```bash
cd hackathon
cat 04_student_train_images.tar.part?? | tar xf -
# -> student_docker/imagenet_released/train/ 생성됨
```

### 방법 2 — 자체 ImageNet-1k(ILSVRC2012) 사본에서 복원

`released_filelist.txt`(113,566줄)가 필요한 파일을 정확히 지정합니다.
경로 형식은 표준 ImageNet train 구조와 동일합니다.

```
train/n10565667/n10565667_7759.JPEG
train/n10565667/n10565667_7576.JPEG
...
```

```bash
# <IMAGENET_ROOT>는 보유 중인 ILSVRC2012 루트 경로
while read -r p; do
  mkdir -p "student_docker/imagenet_released/$(dirname "$p")"
  cp "<IMAGENET_ROOT>/$p" "student_docker/imagenet_released/$p"
done < released_filelist.txt
```

복원 후 파일 수가 정확히 **113,566개**인지 확인하세요.

```bash
find student_docker/imagenet_released -name '*.JPEG' | wc -l   # 113566
```

> 방법 2를 쓸 경우 ILSVRC2012 원본이어야 합니다. 다른 버전/재인코딩본은
> 채점 서버의 사전계산 특징(`score_cache/refs.pt`)과 어긋나 **점수가 조용히
> 틀어질 수 있습니다.** 확신이 없으면 방법 1을 사용하세요.
