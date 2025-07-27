# Use slim Python 3.12 base image
FROM python:3.12-slim

# Create a directory for models inside the container
RUN mkdir -p /app/models


# Set working directory inside container to /app
WORKDIR /app

# Copy requirements.txt to working directory
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt


# Copy all source code files into the container
COPY . .


# Expose port 8510 for the FastAPI server
EXPOSE 8510

# Run the FastAPI server with Uvicorn, listen on all interfaces, port 8510
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8510"]