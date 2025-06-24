from fastapi import FastAPI
from src.api.routes import data_routes

app = FastAPI(
    title="MacroScope API",
    description="API for fetching economic and financial data.",
    version="1.0.0"
)

app.include_router(data_routes.router, prefix="/api", tags=["Data"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the MacroScope API"}
