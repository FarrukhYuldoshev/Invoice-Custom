from uuid import UUID
from typing import List, Annotated
from fastapi import APIRouter, Depends, Path
from sqlalchemy.ext.asyncio import AsyncSession
from core.config import db_utils
from . import crud
from .schemas import DataSchema, ResponseSchema

router = APIRouter(prefix="/exam-operator/examinees", tags=["Exam Operator"])


@router.get("/{rfid}", response_model=ResponseSchema)
async def get_exam_operator_examinees(
    rfid: str, session: AsyncSession = Depends(db_utils.session_dependency)
):
    return await crud.get_exam(rfid, session)
