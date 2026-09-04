"""Trusted participant-side validation and submission API."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
app = FastAPI(
    title="Unlearning 로컬 검증·제출 API",
    description=(
        "선택한 models/*.pt를 submission ID별로 보관하고 중앙 private-test "
        "채점 큐에 업로드합니다. 공개 validation 점수는 터미널에서만 확인합니다."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)
app.include_router(router, prefix="/api")


@app.get("/")
def root():
    return {
        "message": "Unlearning 로컬 검증·제출 API",
        "endpoints": {
            "POST /api/validate": "체크포인트 구조 검사",
            "POST /api/submit": "선택한 model.pt → 중앙 private test 큐",
            "GET /api/submissions/<UUID>": "중앙 채점 상태·점수 조회",
        },
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
