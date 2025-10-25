from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from model import NaiveBayes, LinearRegression


# FastAPI Instance
app = FastAPI()
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code = 500,
        content={"message": f"An error occurred: {str(exc)}"}
    )

# User request validation
class Request(BaseModel):
    input: str
    model: str

# Middleware to log HTTP request
#@app.middleware("http")
#async def log_requests(request: Request, callnext):


