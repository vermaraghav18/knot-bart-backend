# Use the official lightweight Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy your backend files into the image
COPY . .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port your app runs on
EXPOSE 10002

# Command to run the BART server
CMD ["uvicorn", "main_bart:app", "--host", "0.0.0.0", "--port", "10002"]
