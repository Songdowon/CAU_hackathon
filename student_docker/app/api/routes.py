import secrets

from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core import config
from app.schemas.schemas import (
    SubmitRequest,
    SubmitResponse,
    ValidateRequest,
    ValidateResponse,
)
from app.services.local_submission import (
    DeliveryPending,
    DeliveryRejected,
    SubmissionBusy,
    get_remote_submission,
    selected_model_path,
    submit_selected_checkpoint,
)
from app.services.submissions import (
    SnapshotConflict,
    SnapshotNotFound,
    SourceChanged,
    UnsafeSource,
)
from app.services.validation import check_submission


api_key_header = APIKeyHeader(name=config.API_KEY_HEADER_NAME, auto_error=False)
router = APIRouter()


def verify_api_key(provided_key: str | None = Security(api_key_header)) -> None:
    try:
        expected_key = config.student_api_key()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if provided_key is None or not secrets.compare_digest(provided_key, expected_key):
        raise HTTPException(status_code=401, detail="API 키가 없거나 올바르지 않습니다.")


@router.post(
    "/validate",
    response_model=ValidateResponse,
    dependencies=[Depends(verify_api_key)],
)
def validate_model(request: ValidateRequest):
    """Check that one named experiment loads with the official model."""
    try:
        return check_submission(
            str(selected_model_path(request.model_filename)),
            config.NUM_CLASSES,
        )
    except SnapshotNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (UnsafeSource, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"체크포인트 로드에 실패했습니다: {exc}"
        ) from exc


@router.post(
    "/submit",
    response_model=SubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def submit_model(request: SubmitRequest) -> SubmitResponse:
    """Snapshot one named experiment and upload it to the private-test queue."""
    try:
        expected_password = config.submit_password()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if not secrets.compare_digest(request.submit_password, expected_password):
        raise HTTPException(status_code=401, detail="제출 비밀번호가 올바르지 않습니다.")

    try:
        result = submit_selected_checkpoint(request.model_filename)
    except SubmissionBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SnapshotNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (SnapshotConflict, SourceChanged, UnsafeSource, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DeliveryPending as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except DeliveryRejected as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return SubmitResponse.model_validate(result)


@router.get("/submissions/{submission_id}")
def submission_status(submission_id: str):
    """Proxy the authenticated private-test status without exposing its token."""
    try:
        return get_remote_submission(submission_id)
    except (ValueError, DeliveryRejected) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DeliveryPending as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
