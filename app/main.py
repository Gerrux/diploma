import uvicorn
from fastapi import FastAPI

from app.routes import router
import torch

torch.cuda.empty_cache()

app = FastAPI(title="module")


async def startup():
    torch.cuda.empty_cache()


@app.get("/")
async def root():
    return {"message": "Welcome to SecureSight Assistant API"}


# Добавляем контекст жизненного цикла к приложению
app.add_event_handler("startup", startup)
app.include_router(router)
