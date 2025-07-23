
# Base image with Python
FROM python:3.12-slim


WORKDIR /app

# העתקת קובץ requirements.txt.txt לקונטיינר
COPY requirements.txt .

# התקנת תלותות פייתון
RUN pip install --upgrade pip && pip install -r requirements.txt

# העתקת קוד המקור של הסרבר לקונטיינר
COPY . .

# פתיחת פורט 8000 (הפורט שהסרבר רץ עליו)
EXPOSE 8009

# הפקודה להרצת הסרבר עם Uvicorn, מאזין לכל הכתובות, פורט 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8009"]