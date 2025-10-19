from uuid import UUID
from typing import List, Annotated

from fastapi import APIRouter, Depends, Path
from sqlalchemy.ext.asyncio import AsyncSession
from core.config import db_utils
from . import crud
from api_v1.invoice.schemas import GetInvoice, CreateInvoice

router = APIRouter(prefix="/invoice", tags=["invoice"])


@router.post("/", response_model=GetInvoice)
async def create_invoice(
    data: CreateInvoice = Depends(CreateInvoice),
    session: AsyncSession = Depends(db_utils.session_dependency),
):
    return await crud.create_invoice(invoice=data, session=session)


@router.get("/", response_model=List[GetInvoice])
async def get_all_invoices(
    session: AsyncSession = Depends(db_utils.session_dependency),
):
    return await crud.get_all_invoices(session)


@router.get("/{uuid}", response_model=GetInvoice)
async def get_invoice(
    uuid: UUID = Path(...),
    session: AsyncSession = Depends(db_utils.session_dependency),
):
    return await crud.get_invoice(_uuid=uuid, session=session)
