FROM python:3.10-slim

WORKDIR /app

# Install dependencies first to leverage caching
COPY requirements.txt .
RUN apt-get update && apt-get install -y gcc && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get remove -y gcc && apt-get autoremove -y && apt-get clean

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main_bart:app", "--host", "0.0.0.0", "--port", "10002"]
