import datetime
from sqlalchemy import Integer, String, DateTime, Boolean
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base


class Token(Base):
    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, nullable=True, autoincrement=True, index=True
    )
    token: Mapped[str] = mapped_column(String(), nullable=True)
    expires: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False)
