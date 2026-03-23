FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Cloud Run listens on PORT env var (default 8080)
ENV PORT=8080

CMD ["python", "bot.py"]
