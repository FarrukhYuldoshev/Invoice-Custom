from typing import Optional, Annotated, Dict, List
from pydantic import BaseModel, HttpUrl
import datetime
from uuid import UUID
import enum


class StatusEnum(str, enum.Enum):
    ontime = "ontime"
    early = "early"
    late = "late"


class Message(enum.Enum):
    ontime = "Желаем вам удачи на экзаменах!"
    early = "Вы пришли на экзамен раньше назначенного времени"
    late = "Вы опоздали на экзамен."


class Examinees(BaseModel):
    username: str
    password: str
    subject: str
    allowed: bool
    date: datetime.date
    time: datetime.time
    status: StatusEnum


class StudentSchema(BaseModel):
    id: Optional[UUID] = None
    identifier: Optional[str] = None
    photo: Optional[HttpUrl] = None
    fullName: Optional[str] = None


class DataSchema(BaseModel):
    student: StudentSchema
    examinees: List[Examinees]


class Details(BaseModel):
    current_date: datetime.datetime = datetime.datetime.now()
    room: str
    status: str = "Желаем вам удачи на экзаменах!"


class StudentExaminesResponse(BaseModel):
    status: str
    data: DataSchema
    details: Details
