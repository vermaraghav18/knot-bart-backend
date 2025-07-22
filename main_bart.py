from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import uvicorn
import os
import traceback

app = FastAPI()

# Load lightweight summarizer (under 512MB)
model_name = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# ----------- Models ------------
class SummaryRequest(BaseModel):
    text: str
    sentences: int = 5
    lang: str = "en"
    title: str = ""

class ErrorResponse(BaseModel):
    error_type: str
    detail: str
    traceback: str | None = None

# ----------- Utils ------------
def clean_html(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    return ' '.join(soup.get_text(separator=' ').split())

def limit_words(text: str, max_words=120) -> str:
    words = text.split()
    return ' '.join(words[:max_words]) + ('...' if len(words) > max_words else '')

def is_similar(summary: str, title: str, threshold: float = 0.85) -> bool:
    return title.lower().strip() in summary.lower().strip()

# ----------- Global Error Handler ------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb_str = ''.join(traceback.format_tb(exc.__traceback__)) if exc.__traceback__ else None
    error_response = ErrorResponse(
        error_type=type(exc).__name__,
        detail=str(exc),
        traceback=tb_str,
    )
    return JSONResponse(status_code=500, content=error_response.dict())

# ----------- API Endpoint ------------
@app.post("/summarize")
async def summarize(request: SummaryRequest):
    clean_text = clean_html(request.text)
    word_count = len(clean_text.split())

    # 🛑 Block overly long inputs
    if len(clean_text) > 2048:
        return JSONResponse(
            status_code=413,
            content={"error": "Input text too long. Please reduce the content."}
        )

    # 🔹 Short input → return capped original
    if word_count < 150:
        capped = limit_words(clean_text, max_words=120)
        return JSONResponse(content={"summary": [capped]})

    try:
        input_ids = tokenizer.encode("summarize: " + clean_text, return_tensors="pt", max_length=512, truncation=True)
        output_ids = model.generate(input_ids, max_length=80, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

        summary_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        if is_similar(summary_text, request.title):
            summary_text = "Summary too similar to title. Skipped."

        capped_summary = limit_words(summary_text, max_words=120)

        return JSONResponse(
            content={"summary": [capped_summary]},
            media_type="application/json; charset=utf-8"
        )

    except Exception as e:
        raise RuntimeError(f"Summarization failed: {str(e)}")

# ----------- Entry Point ------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10002))
    uvicorn.run("main_bart:app", host="0.0.0.0", port=port)
