from typing import Dict, Any

from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import UUID, text, DATETIME, func, String
from .base import Base
from uuid import uuid4


class Invoice(Base):
    uuid: Mapped[str] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        primary_key=True,
        default=uuid4,
        server_default=text("uuid_generate_v4()"),
    )
    data: Mapped[Dict[str, Any]] = mapped_column(
        MutableDict.as_mutable(JSONB), nullable=False, default=dict
    )
    file: Mapped[str] = mapped_column(String(100), nullable=True)
