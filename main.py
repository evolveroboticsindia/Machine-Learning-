from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():

    return {
        "message": "Stock Sense API Running"
    }