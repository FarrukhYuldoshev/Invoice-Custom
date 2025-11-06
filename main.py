from contextlib import asynccontextmanager

from fastapi import FastAPI
import uvicorn
from sqlalchemy.ext.asyncio import AsyncSession

from api_v1 import exam_operator_router
from core.config import db_utils
from openpyxl import load_workbook
from datetime import datetime, date, time
from typing import List
from dataclasses import dataclass
from core.models import Student


# if we need some actions in starting application
@asynccontextmanager
async def lifespan(app: FastAPI):
    @dataclass
    class StudentExcel:
        full_name: str
        student_id: str
        login: str
        password: str
        subject: str
        exam_date: date
        exam_time: time
        room: str
        rfid: str

    def read_students_from_excel(file_path: str) -> List[StudentExcel]:
        """
        Excel fayldan student ma'lumotlarini o'qiydi.

        Column mapping:
        - full_name: B ustuni (B5 dan boshlab)
        - student_id: C ustuni (C5 dan boshlab)
        - login: D ustuni (D5 dan boshlab)
        - password: E ustuni (E5 dan boshlab)
        - subject: F ustuni (F5 dan boshlab)
        - exam_date: G ustuni (G5 dan boshlab)
        - exam_time: H ustuni (H5 dan boshlab)
        - room: I ustuni (I5 dan boshlab)
        - rfid: J ustuni (J5 dan boshlab)
        """

        # Excel faylni yuklash
        workbook = load_workbook(file_path)
        sheet = workbook.active  # Birinchi varaqni olish

        students = []
        row = 5  # 5-qatordan boshlaymiz

        # Bo'sh qatorga yetguncha o'qiymiz
        while True:
            # Agar B ustuni (full_name) bo'sh bo'lsa, to'xtaymiz
            if sheet[f"B{row}"].value is None:
                break

            # Ma'lumotlarni o'qish
            full_name = str(sheet[f"A{row}"].value or "")
            student_id = str(sheet[f"B{row}"].value or "")
            login = str(sheet[f"C{row}"].value or "")
            password = str(sheet[f"D{row}"].value or "")
            subject = str(sheet[f"E{row}"].value or "")

            # Sana va vaqtni o'qish
            exam_date_value = sheet[f"F{row}"].value
            exam_time_value = sheet[f"G{row}"].value

            # Sanani date obyektiga o'tkazish
            if isinstance(exam_date_value, datetime):
                exam_date = exam_date_value.date()
            elif isinstance(exam_date_value, date):
                exam_date = exam_date_value
            else:
                # String bo'lsa parse qilish
                exam_date = datetime.strptime(str(exam_date_value), "%Y-%m-%d").date()

            # Vaqtni time obyektiga o'tkazish
            if isinstance(exam_time_value, datetime):
                exam_time = exam_time_value.time()
            elif isinstance(exam_time_value, time):
                exam_time = exam_time_value
            else:
                # String bo'lsa parse qilish
                exam_time = datetime.strptime(str(exam_time_value), "%H:%M").time()

            room = str(sheet[f"H{row}"].value or "")
            rfid = str(sheet[f"I{row}"].value or "")

            # Student obyektini yaratish
            student = StudentExcel(
                full_name=full_name,
                student_id=student_id,
                login=login,
                password=password,
                subject=subject,
                exam_date=exam_date,
                exam_time=exam_time,
                room=room,
                rfid=rfid,
            )

            students.append(student)
            row += 1

        workbook.close()
        return students

    # Excel fayldan o'qish
    file_path = "examcontroldata_backup.xlsx"  # Sizning Excel faylingiz nomi
    try:
        students_list = read_students_from_excel(file_path)
        print(f"Jami {len(students_list)} ta student topildi")

    except FileNotFoundError:
        print(f"Fayl topilmadi: {file_path}")
    except Exception as e:
        print(f"Xatolik yuz berdi: {e}")
    async with db_utils.session_factory() as session:  # type: AsyncSession
        for student in students_list:  # type: StudentExcel
            st = Student(**student.__dict__)
            session.add(st)
        await session.commit()
        # session.close()
    yield


app = FastAPI()
app.include_router(exam_operator_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
