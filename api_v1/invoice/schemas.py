from typing import Annotated, Any
from pydantic import BaseModel, field_validator, Field, ConfigDict
from fastapi import Form, UploadFile, File, HTTPException
from uuid import UUID
from datetime import datetime


class CreateInvoice:
    def __init__(self, file: Annotated[UploadFile, File(...)]):
        self.file = self.validate_file_type(file)

    def validate_file_type(cls, file: UploadFile):
        allowed_types = [
            "application/pdf",
            "image/jpeg",
            "image/jpg",
            "image/png",
            "image/webp",
        ]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Only {allowed_types} was supporting. Uploaded file type: {file.content_type}",
            )
        if file.size > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"The maximum file size is 10MB",
            )
        return file


class GetInvoice(BaseModel):
    uuid: Annotated[UUID, Field()]
    file: str
    hashsum: str
    created_at: datetime
    updated_at: datetime
    data: dict[str, Any]
    model_config = ConfigDict(from_attributes=True)
