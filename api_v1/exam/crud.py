from datetime import datetime, timedelta
import uuid
from pathlib import Path
from typing import List

from fastapi import HTTPException
from sqlalchemy import insert, select, ScalarResult
from sqlalchemy.ext.asyncio import AsyncSession
from .schemas import (
    StudentSchema,
    Examinees,
    StatusEnum,
    DataSchema,
    ResponseSchema,
)
from core.models import Student


async def get_exam(rfid: str, session: AsyncSession):
    now = datetime.now()
    stmt = select(Student).where(Student.rfid == rfid).order_by(Student.exam_time)
    result = list(await session.scalars(stmt))
    if result is None or len(result) == 0:
        raise HTTPException(
            status_code=404,
            detail="Not found student",
        )
    else:
        student = result[0]
        studentSchema = StudentSchema(
            id=student.uuid,
            fullName=student.full_name,
            photo=None,
            identifier=student.student_id,
        )
        examinees: List[Examinees] = []

        for exam in result:  # type: Student
            status_enum = StatusEnum.late
            dt = datetime.combine(datetime.today(), exam.exam_time)
            print(dt)
            print(now)
            print(now < dt - timedelta(minutes=20))
            if now >= dt - timedelta(minutes=20) and now <= dt + timedelta(minutes=6):
                status_enum = StatusEnum.ontime
            elif now < dt - timedelta(minutes=20):
                status_enum = StatusEnum.early
            else:
                status_enum = StatusEnum.late
            examSchema = Examinees(
                username=exam.login,
                password=exam.password,
                subject=exam.subject,
                allowed=True,
                date=exam.exam_date,
                time=exam.exam_time,
                status=status_enum,
            )
            examinees.append(examSchema)
    dataSchema = DataSchema(student=studentSchema, examinees=examinees)
    response = ResponseSchema(data=dataSchema)
    return response
