from fastapi import FastAPI

app = FastAPI(
    title="Sales Agent API",
    version="0.1.0"
)


@app.get("/")
def root():
    return {
        "message": "Sales Agent Backend Running"
    }
