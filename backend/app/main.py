from fastapi import FastAPI

from app.api.activities import router as activities_router

app = FastAPI(
    title="Sales Agent API",
    version="0.1.0"
)

app.include_router(activities_router)


@app.get("/")
def root():
    return {
        "message": "Sales Agent Backend Running"
    }
