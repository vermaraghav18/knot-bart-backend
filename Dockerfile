# Use a minimal image with Python 3.10
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy everything into the container
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Render
EXPOSE 10002

# Required environment for models like mT5
ENV TOKENIZERS_PARALLELISM=false

# Start the app (Render injects $PORT)
CMD ["uvicorn", "main_bart:app", "--host", "0.0.0.0", "--port", "10002"]
