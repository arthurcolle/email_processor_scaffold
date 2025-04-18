FROM python:3.11-slim

WORKDIR /app

# Install SQLite and other dependencies
RUN apt-get update && apt-get install -y \
    sqlite3 \
    libsqlite3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install sqlalchemy

COPY . .

# Create data directory for SQLite database
RUN mkdir -p /app/data
ENV DATABASE_URL="sqlite:////app/data/emails.db"

# Set default environment variables
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV FAKEMAIL_API_URL=http://fakemail-api:8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005", "--reload"] 