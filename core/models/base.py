import datetime

from sqlalchemy.orm import DeclarativeBase, declared_attr, Mapped, mapped_column
from sqlalchemy import DateTime, func, text


class Base(DeclarativeBase):
    __abstract__ = True

    @declared_attr.directive
    def __tablename__(cls) -> str:
        return cls.__name__.lower() + "s"

    @declared_attr
    def created_at(self) -> Mapped[datetime.datetime]:
        return mapped_column(
            DateTime,
            nullable=False,
            server_default=text("current_timestamp"),
            default=datetime.datetime.now,
        )

    @declared_attr
    def updated_at(self) -> Mapped[datetime.datetime]:
        return mapped_column(
            DateTime,
            default=datetime.datetime.now,
            onupdate=datetime.datetime.now,
            server_default=func.now(),
            server_onupdate=func.now(),
            nullable=False,
        )

    def __repr__(self) -> str:
        cols: list = []
        for column in self.__table__.columns.keys():
            cols.append(f"{column} = {getattr(self, column)!r}")
        return f"{self.__class__.__name__}: ({'; '.join(cols)})"
