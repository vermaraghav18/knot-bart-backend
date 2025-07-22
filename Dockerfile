FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Download punkt tokenizer needed by Sumy
RUN python -m nltk.downloader punkt

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main_bart:app", "--host", "0.0.0.0", "--port", "10002"]
