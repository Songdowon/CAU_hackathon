from typing import Literal

from pydantic import BaseModel, Field


class ValidateResponse(BaseModel):
    status: str
    loads_cleanly: bool
    logits_shape: list[int]
    num_parameters: int
    message: str


class ValidateRequest(BaseModel):
    model_filename: str = Field(min_length=4, max_length=203)


class SubmitRequest(BaseModel):
    submit_password: str = Field(min_length=1, max_length=512)
    model_filename: str = Field(min_length=4, max_length=203)


class SubmitResponse(BaseModel):
    status: Literal["queued", "running", "done", "error"]
    submission_id: str
    team_name: str
    model_filename: str
    artifact_sha256: str
    artifact_size_bytes: int
    snapshot_path: str
    remaining_attempts: int
