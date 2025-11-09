from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from model import SpamClassifier
from utils import Logger
import time

# FastAPI Instance
app = FastAPI()

# Logger Instance
logger = Logger.setup_logger()

# CORS handle middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Handle exception
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code = 500,
        content={"message": f"An error occurred: {str(exc)}"}
    )

# User request validation
# class UserInput(BaseModel):
#     model: str = Form(default='NaiveBayes', description="Model selected to predict the email text")

# Log HTTP request
@app.middleware("http")
async def log_requests(request: Request, callnext):
    start_time = time.time()
    response = await callnext(request)
    process_time = time.time() - start_time
    print(f"Request: {request.url} - Duration: {process_time} seconds")
    return response

# Landdle exception
@app.exception_handler(HTTPException)
async def http_except_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code = exc.status_code,
        content={"detail": exc.detail, "error": "An error has occurred"}
    )

# Initilise Models
# nb = NaiveBayes()
# lg = LinearRegression()
classifier = SpamClassifier()

# Root call
@app.get("/")
async def root():
    return {"message": "Hello welcome to spam detection app"}

# POST endpoint at "/predict"
@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = Form(default='NaiveBayes', description="Model selected to predict the email text")):
    try:
        # Call the model's predict method using the input data
        if not file.filename.endswith('.csv'):
           return {"error": "File must be a CSV"}
        
        result = classifier.spam_classify(file.file, model)
        
        logger.info(model)

        # Log the prediction details
        logger.info(f"Prediction came out as: {result["spam_count"]} spam emails for \"{file.filename}\" text, using {model}")
        
        # Return the predicted email type in JSON format

        return JSONResponse(
           content=result
        )
    except Exception as e:
        # Log the error if an exception occurs during prediction
        logger.error(f"Error during prediction: {str(e)}")
        
        # Raise an HTTP 500 Internal Server Error if prediction fails
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("__main__:app", host="0.0.0.0", port=8000, reload=True)