FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY calculate.py .

ENV CALCULATE_CMD="python calculate.py"