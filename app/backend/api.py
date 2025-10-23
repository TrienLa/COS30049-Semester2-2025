from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


# FastAPI Instance
app = FastAPI()

# Middleware to log HTTP request
#@app.middleware("http")
#async def log_requests(request: Request, callnext):


