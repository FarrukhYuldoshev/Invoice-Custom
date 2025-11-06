from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import UUID, text, String, UniqueConstraint
from .base import Base
from uuid import uuid4
import datetime


class Student(Base):
    uuid: Mapped[str] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        primary_key=True,
        default=uuid4,
        server_default=text("uuid_generate_v4()"),
    )
    full_name: Mapped[str] = mapped_column(String(), nullable=False)
    student_id: Mapped[str] = mapped_column(String(256), nullable=False)
    login: Mapped[str] = mapped_column(String(128), nullable=False)
    password: Mapped[str] = mapped_column(String(128), nullable=False)
    subject: Mapped[str] = mapped_column(String(), nullable=False)
    exam_date: Mapped[datetime.date] = mapped_column(nullable=False)
    exam_time: Mapped[datetime.time] = mapped_column(nullable=False)
    room: Mapped[str] = mapped_column(String(), nullable=False)
    rfid: Mapped[str] = mapped_column(String(), nullable=True)
    __table_args__ = (
        UniqueConstraint(
            "student_id",
            "subject",
            "rfid",
            name="uix_student_id_subject_rfid",
        ),
    )
