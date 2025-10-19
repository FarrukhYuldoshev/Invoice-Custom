from datetime import datetime
import uuid
from pathlib import Path
from fastapi import HTTPException
from sqlalchemy import insert, select, ScalarResult
from sqlalchemy.ext.asyncio import AsyncSession
from api_v1.invoice.schemas import CreateInvoice
from core.models import Invoice
import hashlib

UPLOAD_DIR = Path("static/upload/invoices")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def calculate_file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


async def create_invoice(
    invoice: CreateInvoice, session: AsyncSession
) -> Invoice | None:
    file = invoice.file
    allowed_types = {
        "application/pdf": ".pdf",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }
    file_extension = allowed_types[file.content_type]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = UPLOAD_DIR / unique_filename
    try:
        content = await file.read()
        hashsum = calculate_file_hash(content)
        stmt = select(Invoice).where(Invoice.hashsum == hashsum)
        found_invoice_with_hash: Invoice = await session.scalar(stmt)
        if found_invoice_with_hash:
            found_invoice_with_hash.data = {
                "details": f"New updated data invoice. Time:{datetime.now()}"
            }
            await session.commit()
            return found_invoice_with_hash

        with open(file_path, "wb") as f:
            f.write(content)
        inv: Invoice = Invoice(
            file=str(file_path), data={"details": "This is an invoice"}, hashsum=hashsum
        )
        session.add(inv)
        await session.commit()
        return inv
    except Exception as e:
        if Path(file_path).exists():
            Path(file_path).unlink()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        await file.seek(0)


async def get_invoice(_uuid: uuid.UUID, session: AsyncSession) -> Invoice | None:
    stmt = select(Invoice).where(Invoice.uuid == _uuid)
    inv = await session.scalar(stmt)
    if not inv:
        raise HTTPException(status_code=404, detail="Invoice not found")
    return inv


async def get_all_invoices(session: AsyncSession) -> list[Invoice] | None:
    stmt = select(Invoice)
    invoices = await session.scalars(stmt)
    return list(invoices)
