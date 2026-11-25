FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
COPY pyproject.toml .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and artifacts
COPY src/ src/
COPY api/ api/
COPY configs/ configs/
COPY checkpoints/ checkpoints/
COPY results/ results/
RUN pip install --no-cache-dir -e .

ENV PORT=8000
ENV DEVICE=cpu
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
