# Use the official lightweight Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy your backend files into the image
COPY . .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the dynamic port from Render (default fallback to 10002)
EXPOSE 10002

# Use shell form so env vars like $PORT are interpreted
CMD uvicorn main_bart:app --host 0.0.0.0 --port ${PORT:-10002}
