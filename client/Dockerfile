# Base image with Python
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# העתק את קובץ הדרישות
COPY requirements.txt .

# התקן את התלויות
RUN pip install --no-cache-dir -r requirements.txt

# העתק את כל הקוד של הקליינט
COPY . .

# Expose port for Streamlit
EXPOSE 8501

# הפקודה להרצת אפליקציית Streamlit

CMD ["streamlit", "run", "app.py", "--server.port=8011", "--server.address=0.0.0.0"]