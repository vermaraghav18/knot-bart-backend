from fastapi import FastAPI, Request
import os
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from bs4 import BeautifulSoup
import uvicorn
import traceback

app = FastAPI()

# ✅ Use a light T5 summarization model
model_name = "Falconsai/text-summarization-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

class SummaryRequest(BaseModel):
    text: str
    sentences: int = 5
    lang: str = "en"
    title: str = ""

class ErrorResponse(BaseModel):
    error_type: str
    detail: str
    traceback: str | None = None

def clean_html(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    return ' '.join(soup.get_text(separator=' ').split())

def limit_words(text: str, max_words=120) -> str:
    words = text.split()
    return ' '.join(words[:max_words]) + ('...' if len(words) > max_words else '')

def is_similar(summary: str, title: str, threshold: float = 0.85) -> bool:
    return title.lower().strip() in summary.lower().strip()

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb_str = ''.join(traceback.format_tb(exc.__traceback__)) if exc.__traceback__ else None
    error_response = ErrorResponse(
        error_type=type(exc).__name__,
        detail=str(exc),
        traceback=tb_str,
    )
    return JSONResponse(status_code=500, content=error_response.dict())

@app.post("/summarize")
async def summarize(request: SummaryRequest):
    clean_text = clean_html(request.text)
    word_count = len(clean_text.split())

    if word_count < 150:
        capped = limit_words(clean_text, max_words=120)
        return JSONResponse(content={"summary": [capped]})

    # Light T5 summarizer call
    result = summarizer(clean_text, max_length=80, min_length=30, do_sample=False)
    summary_text = result[0]['summary_text']

    if is_similar(summary_text, request.title):
        summary_text = "Summary too similar to title. Skipped."

    capped_summary = limit_words(summary_text, max_words=120)

    return JSONResponse(
        content={"summary": [capped_summary]},
        media_type="application/json; charset=utf-8"
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10002))
    uvicorn.run("main_bart:app", host="0.0.0.0", port=port)
