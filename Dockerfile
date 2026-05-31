FROM python:3.10-slim

WORKDIR /app

# Install dependencies first to maximize Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy runtime artifacts only.
COPY alembic.ini .
COPY alembic/ ./alembic/
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
