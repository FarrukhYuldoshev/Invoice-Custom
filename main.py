from contextlib import asynccontextmanager

from fastapi import FastAPI
import uvicorn

app = FastAPI()


# if we need some actions in starting application
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
