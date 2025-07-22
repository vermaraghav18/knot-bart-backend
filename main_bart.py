from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from bs4 import BeautifulSoup
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import traceback
import os
import uvicorn

app = FastAPI()

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

def sumy_summarize(text: str, sentence_count: int = 5) -> str:
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary_sentences = summarizer(parser.document, sentence_count)
    return ' '.join(str(sentence) for sentence in summary_sentences)

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

    if word_count < 50:
        return JSONResponse(content={"summary": [clean_text]})

    try:
        summary = sumy_summarize(clean_text, sentence_count=request.sentences)
        return JSONResponse(content={"summary": [summary]})
    except Exception as e:
        raise RuntimeError(f"Summarization failed: {str(e)}")

# ----------- Entry Point ------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10002))
    uvicorn.run("main_bart:app", host="0.0.0.0", port=port)
