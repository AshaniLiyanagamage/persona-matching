from fastapi import FastAPI
from routes import router
from fastapi.responses import HTMLResponse

app = FastAPI()

app.include_router(router)
