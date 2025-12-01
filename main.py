from fastapi import FastAPI
"""This file has the only purpose of checking FastAPI installation."""
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}