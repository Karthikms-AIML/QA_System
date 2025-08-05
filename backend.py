from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ["http://127.0.0.1:5500"] for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pretrained QA model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

class QARequest(BaseModel):
    context: str
    question: str

@app.post("/predict")
def predict(data: QARequest):
    try:
        result = qa_pipeline(question=data.question, context=data.context)
        return {"answer": result["answer"]}
    except Exception as e:
        return {"error": str(e)}
